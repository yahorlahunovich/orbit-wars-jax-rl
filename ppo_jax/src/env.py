from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any

from env.jax_orbit_wars import JaxEnvState, jax_orbit_wars_reset, jax_orbit_wars_step
from .orbit_wars.features_jax import ObsBatch, extract_obs_v9_jax


def compute_reward(
    state_before: JaxEnvState,
    state_after: JaxEnvState,
    player_id: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    """Computes shaped reward for a player after one step."""
    cur_t = state_after.cur_turn
    p_owners = state_after.future_timeline[:, cur_t, 0]
    p_active = state_after.active_mask

    p0_alive = jnp.any((p_owners == 0) & p_active)
    p1_alive = jnp.any((p_owners == 1) & p_active)

    player_alive = jnp.where(player_id == 0, p0_alive, p1_alive)
    opponent_alive = jnp.where(player_id == 0, p1_alive, p0_alive)

    newly_done = state_after.done & ~state_before.done
    win_reward = newly_done & player_alive & ~opponent_alive
    loss_reward = newly_done & ~player_alive

    win_val = float(cfg.get("win_reward", 1.0))
    loss_val = float(cfg.get("loss_reward", -1.0))
    terminal_r = jnp.where(win_reward, win_val,
                           jnp.where(loss_reward, loss_val, 0.0))

    def potential(s: JaxEnvState) -> jnp.ndarray:
        t = s.cur_turn
        owners = s.future_timeline[:, t, 0]
        ships = s.future_timeline[:, t, 1]
        active = s.active_mask
        player_owned = (owners == player_id) & active
        opp_owned = (owners >= 0) & (owners != player_id) & active
        own_ships = jnp.sum(jnp.where(player_owned, ships, 0.0))
        opp_ships = jnp.sum(jnp.where(opp_owned, ships, 0.0))
        own_prod = jnp.sum(jnp.where(player_owned, s.planet_production, 0.0))
        opp_prod = jnp.sum(jnp.where(opp_owned, s.planet_production, 0.0))
        own_planets = jnp.sum(player_owned.astype(jnp.float32))
        opp_planets = jnp.sum(opp_owned.astype(jnp.float32))
        return (
            0.0010 * (own_ships - opp_ships)
            + 0.0100 * (own_prod - opp_prod)
            + 0.0200 * (own_planets - opp_planets)
        )

    shaping_coef = float(cfg.get("reward_shaping_coef", 0.05))
    shaped_delta = jnp.clip(potential(state_after) - potential(state_before), -0.05, 0.05)
    shaped_r = shaping_coef * shaped_delta
    return terminal_r + jnp.where(newly_done, 0.0, shaped_r)


class OrbitWarsPureJaxEnv:
    """A wrapper for Orbit Wars that makes reset and step fully jittable.
    
    It uses jax.pure_callback to call out to the host for python-based 
    reset (which relies on the Kaggle engine/generation) and precomputations.
    """
    def __init__(self, episode_steps: int = 500, ship_speed: float = 6.0):
        self.episode_steps = episode_steps
        self.ship_speed = ship_speed
        self.dummy_state = jax_orbit_wars_reset(0, as_jax=True)

    def reset(self, key: jax.Array) -> tuple[ObsBatch, JaxEnvState]:
        """Jittable reset using pure_callback."""
        seed = jax.random.randint(key, (), 0, 2**31 - 1)
        
        def _host_reset(s: int) -> JaxEnvState:
            return jax_orbit_wars_reset(int(s), as_jax=False)

        state = jax.pure_callback(
            _host_reset,
            self.dummy_state,
            seed,
            vmap_method="sequential",
        )
        obs = extract_obs_v9_jax(state, player_id=0)
        return obs, state

    def step(
        self, 
        key: jax.Array, 
        state: JaxEnvState, 
        actions_p0: jnp.ndarray, 
        owned_p0: jnp.ndarray,
        actions_p1: jnp.ndarray,
        owned_p1: jnp.ndarray,
    ) -> tuple[ObsBatch, JaxEnvState, jnp.ndarray, jnp.ndarray, dict]:
        """Jittable step without callbacks."""
        
        inert_actions = jnp.full(actions_p0.shape, -1, dtype=jnp.int32)
        inert_owned = jnp.full(owned_p0.shape, -1, dtype=jnp.int32)
        
        # 4-player format: shape (4, S) where S is the action size
        actions_4p = jnp.stack([actions_p0, actions_p1, inert_actions, inert_actions], axis=0)
        owned_4p = jnp.stack([owned_p0, owned_p1, inert_owned, inert_owned], axis=0)

        # 1. Pure JAX step
        next_state = jax_orbit_wars_step(
            state,
            actions_4p,
            owned_4p,
        )
        
        # 2. Rewards
        reward = compute_reward(state, next_state, player_id=jnp.array(0, dtype=jnp.int32), cfg={
            "win_reward": 1.0,
            "loss_reward": -1.0,
            "reward_shaping_coef": 0.05,
        })
        
        # Done flag
        done = next_state.done
        
        next_obs = extract_obs_v9_jax(next_state, player_id=0)
        
        # 3. Auto-reset
        def reset_fn():
            return self.reset(key)
            
        def keep_fn():
            return next_obs, next_state
            
        obs_out, state_out = jax.lax.cond(done, reset_fn, keep_fn)
        
        return obs_out, state_out, reward, done, {}
