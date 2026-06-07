"""JAX implementation of the 'Producer Agent' (1230 ELO) logic.
Ported from orbit_lite (PyTorch) to JAX for optimized PPO training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from .state import OrbitWarsState
from .constants import (
    CENTER, 
    SUN_RADIUS, 
    BOARD_SIZE, 
    NUM_PLAYERS,
    MAX_PLANETS,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

class GarrisonStatus(NamedTuple):
    """Post-combat owner and ships over time [P, H+1]."""
    owner: jnp.ndarray
    ships: jnp.ndarray
    pre_combat_owner: jnp.ndarray
    pre_combat_ships: jnp.ndarray
    arrivals_by_owner: jnp.ndarray  # [P, H+1, A]


# ---------------------------------------------------------------------------
# Core Simulation / Prediction
# ---------------------------------------------------------------------------

def _per_step_survivor(arrivals: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combat survivor over player axis: (owner, ships) [..., A] -> (...,)"""
    A = arrivals.shape[-1]
    if A == 2:
        s0 = arrivals[..., 0]
        s1 = arrivals[..., 1]
        top_ships = jnp.maximum(s0, s1)
        second_ships = jnp.minimum(s0, s1)
        top_owner = (s1 > s0).astype(jnp.int32)
    else:
        sorted_arr = jnp.sort(arrivals, axis=-1)
        top_ships = sorted_arr[..., -1]
        second_ships = sorted_arr[..., -2]
        top_owner = jnp.argmax(arrivals, axis=-1)
        
    tied = jnp.equal(top_ships, second_ships)
    survivor_ships = jnp.where(tied, 0.0, top_ships - second_ships)
    return top_owner, survivor_ships


def project_garrison(
    state: OrbitWarsState,
    horizon: int,
    extra_arrivals: jnp.ndarray | None = None, # [P, H, A]
) -> GarrisonStatus:
    """Project planet owner/ships over H steps using exact recurrence."""
    H = horizon
    P = state.planets.shape[0]
    A = NUM_PLAYERS
    
    init_owner = state.planets[:, 1].astype(jnp.int32)
    init_ships = state.planets[:, 5]
    prod = state.planets[:, 6]
    alive = state.planets[:, 7] > 0.0
    
    arrivals = extra_arrivals if extra_arrivals is not None else jnp.zeros((P, H, A))
    
    def step_fn(carry, t):
        curr_owner, curr_ships = carry
        produces = (curr_owner >= 0) & alive
        next_ships = curr_ships + jnp.where(produces, prod, 0.0)
        pre_owner = curr_owner
        pre_ships = next_ships
        
        s_owner, s_ships = _per_step_survivor(arrivals[:, t, :])
        has_combat = (s_ships > 0.0)
        same = (curr_owner == s_owner)
        diff = next_ships - s_ships
        attacker_wins = (~same) & (diff < 0.0)
        
        combat_ships = jnp.where(same, next_ships + s_ships, jnp.abs(diff))
        combat_owner = jnp.where(attacker_wins, s_owner, curr_owner)
        
        next_ships = jnp.where(has_combat, combat_ships, next_ships)
        next_owner = jnp.where(has_combat, combat_owner, curr_owner)
        
        next_ships = jnp.where(alive, next_ships, 0.0)
        next_owner = jnp.where(alive, next_owner, -1)
        
        return (next_owner, next_ships), (next_owner, next_ships, pre_owner, pre_ships)

    _, (o_traj, s_traj, po_traj, ps_traj) = jax.lax.scan(
        step_fn, (init_owner, init_ships), jnp.arange(H)
    )
    
    # Prepend initial state
    # Scan outputs are [H, P]. Transpose to [P, H]
    o_traj = jnp.concatenate([init_owner[:, None], o_traj.transpose(1, 0)], axis=1)
    s_traj = jnp.concatenate([init_ships[:, None], s_traj.transpose(1, 0)], axis=1)
    po_traj = jnp.concatenate([init_owner[:, None], po_traj.transpose(1, 0)], axis=1)
    ps_traj = jnp.concatenate([init_ships[:, None], ps_traj.transpose(1, 0)], axis=1)
    
    arr_full = jnp.concatenate([jnp.zeros((P, 1, A)), arrivals], axis=1)

    return GarrisonStatus(
        owner=o_traj,
        ships=s_traj,
        pre_combat_owner=po_traj,
        pre_combat_ships=ps_traj,
        arrivals_by_owner=arr_full,
    )


# ---------------------------------------------------------------------------
# Scoring / ROI
# ---------------------------------------------------------------------------

def _flow_terms_per_planet(
    status: GarrisonStatus,
    prod: jnp.ndarray, # [P]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate (produced, combat_lost) per planet per player [P, A]."""
    P, H1 = status.owner.shape
    H = H1 - 1
    A = status.arrivals_by_owner.shape[-1]
    
    owner_before = status.owner[:, :H] # [P, H]
    amount = prod[:, None] # [P, 1]
    
    a_idx = jnp.arange(A)
    prod_mask = (owner_before[:, :, None] == a_idx[None, None, :])
    produced = jnp.sum(amount[:, :, None] * prod_mask, axis=1) # [P, A]
    
    arr_k = status.arrivals_by_owner[:, 1:, :] # [P, H, A]
    s_owner, s_ships = _per_step_survivor(arr_k) # [P, H]
    
    is_survivor = (s_owner[:, :, None] == a_idx[None, None, :])
    survived_ships = jnp.where(is_survivor, s_ships[:, :, None], 0.0)
    attacker_lost = jnp.sum(jnp.maximum(0.0, arr_k - survived_ships), axis=1) # [P, A]
    
    prior_owner = status.pre_combat_owner[:, 1:] # [P, H]
    prior_ships = status.pre_combat_ships[:, 1:] # [P, H]
    fights_garrison = (s_ships > 0.0) & (s_owner != prior_owner) & (s_owner >= 0)
    g_loss = jnp.where(fights_garrison, jnp.minimum(prior_ships, s_ships), 0.0)
    
    is_prior = (prior_owner[:, :, None] == a_idx[None, None, :]) & fights_garrison[:, :, None] & (prior_owner[:, :, None] >= 0)
    is_winning_attacker = (s_owner[:, :, None] == a_idx[None, None, :]) & fights_garrison[:, :, None]
    garrison_lost = jnp.sum(g_loss[:, :, None] * (is_prior + is_winning_attacker), axis=1) # [P, A]
    
    return produced, attacker_lost + garrison_lost


def competitive_score(produced: jnp.ndarray, lost: jnp.ndarray, player_id: jnp.ndarray | int) -> jnp.ndarray:
    """ROI: (My Net Delta) - (Sum of Opponent Net Deltas)."""
    # produced, lost can be [P, A] or [A]
    net = produced - lost
    A = net.shape[-1]
    
    if net.ndim == 1:
        me = net[player_id]
        opp = jnp.sum(net) - me
        return me - opp
    else:
        # produced [P, A], player_id is scalar
        me = net[:, player_id]
        opp = jnp.sum(net, axis=-1) - me
        return me - opp


def get_heuristic_roi(state: OrbitWarsState, player_id: int, horizon: int = 20) -> jnp.ndarray:
    """Calculate the total ROI score for a player in a given state."""
    status = project_garrison(state, horizon)
    prod = state.planets[:, 6]
    p_produced, p_lost = _flow_terms_per_planet(status, prod)
    # sum over planets
    total_produced = jnp.sum(p_produced, axis=0)
    total_lost = jnp.sum(p_lost, axis=0)
    return competitive_score(total_produced, total_lost, player_id)


def safe_drain(status: GarrisonStatus, source_idx: jnp.ndarray, player_id: int) -> jnp.ndarray:
    """Calculate the minimum garrison ship count that must be preserved on source planets."""
    ships_traj = status.ships[source_idx, 1:]
    owner_traj = status.owner[source_idx, 1:]
    me_owned = (owner_traj == player_id)
    inf_fill = jnp.full_like(ships_traj, 1e9)
    cap_traj = jnp.where(me_owned & (ships_traj > 0.0), ships_traj, inf_fill)
    min_slack = jnp.min(cap_traj, axis=-1)
    current_ships = status.ships[source_idx, 0]
    return jnp.maximum(0.0, jnp.minimum(min_slack, current_ships))
