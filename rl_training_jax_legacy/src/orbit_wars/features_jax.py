"""Pure-JAX feature encoder for Orbit Wars.

This module converts an `OrbitWarsState` (or a batched `vmap` thereof) into
fixed-shape entity-level features:

- planet features:  (B, MAX_PLANETS, F_PLANET)        plus planet_mask  (B, MAX_PLANETS)
- fleet  features:  (B, MAX_FLEETS,  F_FLEET)         plus fleet_mask   (B, MAX_FLEETS)
- global features:  (B, F_GLOBAL)

All features are player-relative (perspective of `player`) — owner flags, lead
ratios, etc. are flipped automatically for the opposite side.

The encoder is JIT-friendly and vmappable. No NumPy bridge.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_SPAWN_STEPS,
    MAX_COMET_GROUPS,
    MAX_FLEETS,
    MAX_PLANETS,
    ROTATION_RADIUS_LIMIT,
)
from .geometry import fleet_speed
from .state import OrbitWarsState
from .producer import project_garrison

PLANET_FEATURE_DIM = 58
FLEET_FEATURE_DIM = 15
GLOBAL_FEATURE_DIM = 22

# Normalization constants.
SHIPS_LOG_DENOM = jnp.log1p(jnp.float32(5000.0))
FLEET_SHIPS_LOG_DENOM = jnp.log1p(jnp.float32(500.0))
PRODUCTION_DENOM = jnp.float32(5.0)
RADIUS_DENOM = jnp.float32(10.0)
DIST_DENOM = jnp.float32(50.0)
INCOMING_PERP_TOL = jnp.float32(3.0)  # how close a fleet's path must come to a planet to count as "incoming"


def _ships_log_norm(ships: jnp.ndarray) -> jnp.ndarray:
    """log1p(ships) / log1p(5000)."""
    return jnp.log1p(jnp.maximum(ships, 0.0)) / SHIPS_LOG_DENOM


def _fleet_ships_log_norm(ships: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.maximum(ships, 0.0)) / FLEET_SHIPS_LOG_DENOM


def _is_comet_per_planet(state: OrbitWarsState) -> jnp.ndarray:
    """Boolean per-planet flag, vectorized."""
    pids = state.planets[:, 0].astype(jnp.int32)
    cpids = state.comet_planet_ids
    valid = cpids >= 0
    return jnp.any((pids[:, None] == cpids[None, :]) & valid[None, :], axis=-1)


def _is_orbiting_per_planet(state: OrbitWarsState) -> jnp.ndarray:
    init = state.initial_planets
    dx = init[:, 2] - CENTER
    dy = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    radius = state.planets[:, 4]
    return orbit_r + radius < ROTATION_RADIUS_LIMIT


def _fleet_projections(state: OrbitWarsState, player_f: jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate exact destination and ETA for all fleets using physics."""
    planets = state.planets
    fleets = state.fleets
    
    fx = fleets[:, 2]
    fy = fleets[:, 3]
    angle = fleets[:, 4]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    fleet_ships = fleets[:, 6]
    speed = fleet_speed(fleet_ships, state.ship_speed)
    fleet_active = fleets[:, 7] > 0.0

    px = planets[:, 2]
    py = planets[:, 3]
    radius = planets[:, 4]
    planet_active = planets[:, 7] > 0.0

    dx = px[None, :] - fx[:, None]     # (F, P)
    dy = py[None, :] - fy[:, None]     # (F, P)
    
    ux = cos_a[:, None]
    uy = sin_a[:, None]
    
    dot_ru = dx * ux + dy * uy
    
    sp = speed[:, None]
    sp_safe = jnp.maximum(sp, 1e-6)
    t_close_raw = dot_ru / sp_safe
    t_close = jnp.where(dot_ru > 0.0, t_close_raw, 0.0)
    
    cx = fx[:, None] + sp * ux * t_close
    cy = fy[:, None] + sp * uy * t_close
    
    d_close = jnp.sqrt((px[None, :] - cx)**2 + (py[None, :] - cy)**2)
    
    slop = jnp.float32(3.0)
    collides = (t_close > 0.0) & (d_close <= radius[None, :] + slop) & planet_active[None, :]
    
    big = jnp.float32(1e9)
    masked_t_close = jnp.where(collides, t_close, big)
    
    best_t_close = jnp.min(masked_t_close, axis=1) # (F,)
    best_p_idx = jnp.argmin(masked_t_close, axis=1) # (F,)
    
    has_target = fleet_active & (best_t_close < big)
    
    fleet_owner = fleets[:, 1]
    f_is_me = has_target & (fleet_owner == player_f)
    f_is_enemy = has_target & (fleet_owner >= 0.0) & (fleet_owner != player_f)
    
    incoming_me = jnp.zeros(MAX_PLANETS, dtype=jnp.float32)
    incoming_me = incoming_me.at[best_p_idx].add(jnp.where(f_is_me, fleet_ships, 0.0))
    
    incoming_enemy = jnp.zeros(MAX_PLANETS, dtype=jnp.float32)
    incoming_enemy = incoming_enemy.at[best_p_idx].add(jnp.where(f_is_enemy, fleet_ships, 0.0))
    
    eta_me = jnp.full(MAX_PLANETS, big, dtype=jnp.float32)
    eta_me = eta_me.at[best_p_idx].min(jnp.where(f_is_me, best_t_close, big))
    
    eta_enemy = jnp.full(MAX_PLANETS, big, dtype=jnp.float32)
    eta_enemy = eta_enemy.at[best_p_idx].min(jnp.where(f_is_enemy, best_t_close, big))
    
    eta_me_norm = jnp.minimum(eta_me, 100.0) / 100.0
    eta_enemy_norm = jnp.minimum(eta_enemy, 100.0) / 100.0
    
    return incoming_me, incoming_enemy, eta_me_norm, eta_enemy_norm


def _nearest_distance_from_matrix(
    dist_matrix: jnp.ndarray,
    subset_mask: jnp.ndarray,
) -> jnp.ndarray:
    big = jnp.float32(1e6)
    # Exclude self and rows outside the subset.
    self_mask = jnp.eye(MAX_PLANETS, dtype=jnp.bool_)
    masked = jnp.where(subset_mask[None, :] & (~self_mask), dist_matrix, big)
    nearest = jnp.min(masked, axis=-1)
    # Convert "no subset member" sentinel (big) to a finite max distance.
    nearest = jnp.where(nearest >= big, DIST_DENOM * 3.0, nearest)
    return nearest


def _rank_norm_from_matrix(
    smaller_matrix: jnp.ndarray, # (P, P) bool: values[i] > values[j]
    mask: jnp.ndarray,
) -> jnp.ndarray:
    eligible_count = jnp.sum(mask.astype(jnp.float32))
    smaller = jnp.sum(
        smaller_matrix
        & mask[None, :]
        & mask[:, None],
        axis=-1,
    ).astype(jnp.float32)
    denom = jnp.maximum(eligible_count - 1.0, 1.0)
    return jnp.where(mask, smaller / denom, 0.0)


def _comet_remaining_life(state: OrbitWarsState) -> jnp.ndarray:
    """For each planet slot, remaining comet path steps (else 0).

    Vectorized: for each (g, q) comet quad, scatter remaining = path_length -
    (path_index + 1) into the planet slot whose id matches.
    """
    comets = state.comets
    cgpids = comets.planet_ids
    cplens = comets.path_lengths
    cactive = comets.active
    idx_next = comets.path_index + 1

    pids = state.planets[:, 0].astype(jnp.int32)
    active = state.planets[:, 7] > 0.0

    match_gqp = (
        (cgpids[..., None] == pids[None, None, :])
        & active[None, None, :]
        & (cgpids[..., None] >= 0)
    )                                                # (G, 4, P)
    slot_gq = jnp.argmax(match_gqp.astype(jnp.int32), axis=-1)
    has_match_gq = jnp.any(match_gqp, axis=-1)
    remaining_gq = (cplens - idx_next[:, None]).astype(jnp.float32)
    remaining_gq = jnp.maximum(remaining_gq, 0.0)
    valid_gq = has_match_gq & cactive[:, None] & (cgpids >= 0)

    out = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    flat_slot = slot_gq.reshape(-1)
    flat_val = jnp.where(valid_gq.reshape(-1), remaining_gq.reshape(-1), 0.0)
    out = out.at[flat_slot].set(jnp.maximum(out[flat_slot], flat_val))
    return out


def encode_observation(
    state: OrbitWarsState,
    player: jnp.int32 | int,
) -> dict[str, jnp.ndarray]:
    """Encode a single (unbatched) state into entity features for `player`.

    Returns a dict with keys:
        planet_features (P, F_PLANET) float32
        planet_mask     (P,)           bool

    Apply `jax.vmap(encode_observation, in_axes=(0, None))` for batched envs.
    """
    player_f = jnp.float32(player)
    planets = state.planets
    fleets = state.fleets

    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    owner_is_me = active & (owner == player_f)
    owner_is_enemy = active & (owner >= 0.0) & (owner != player_f)
    owner_is_neutral = active & (owner < 0.0)

    ships = planets[:, 5]
    production = planets[:, 6]
    radius = planets[:, 4]
    x = planets[:, 2]
    y = planets[:, 3]

    dx_c = (x - CENTER) / DIST_DENOM
    dy_c = (y - CENTER) / DIST_DENOM
    dist_c = jnp.sqrt(dx_c * dx_c + dy_c * dy_c)

    is_orbiting = _is_orbiting_per_planet(state) & active
    is_comet = _is_comet_per_planet(state) & active

    init = state.initial_planets
    dx0 = init[:, 2] - CENTER
    dy0 = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    initial_angle = jnp.arctan2(dy0, dx0)
    current_angle = initial_angle + state.angular_velocity * state.step.astype(jnp.float32)
    orbit_angle_sin = jnp.where(is_orbiting, jnp.sin(current_angle), 0.0)
    orbit_angle_cos = jnp.where(is_orbiting, jnp.cos(current_angle), 0.0)
    orbit_r_norm = jnp.where(is_orbiting, orbit_r / DIST_DENOM, 0.0)

    incoming_me, incoming_enemy, eta_me_norm, eta_enemy_norm = _fleet_projections(state, player_f)

    # Precompute distance matrix O(P^2) once
    dx_pp = x[:, None] - x[None, :]
    dy_pp = y[:, None] - y[None, :]
    dist_pp = jnp.sqrt(dx_pp * dx_pp + dy_pp * dy_pp)

    nearest_enemy_d_raw = _nearest_distance_from_matrix(dist_pp, owner_is_enemy)
    nearest_friend_d_raw = _nearest_distance_from_matrix(dist_pp, owner_is_me)
    nearest_neutral_d_raw = _nearest_distance_from_matrix(dist_pp, owner_is_neutral)
    nearest_enemy_d = nearest_enemy_d_raw / DIST_DENOM
    nearest_friend_d = nearest_friend_d_raw / DIST_DENOM

    max_speed = state.ship_speed
    time_norm = jnp.float32(100.0)
    time_to_nearest_enemy = jnp.where(active, nearest_enemy_d_raw / max_speed / time_norm, 0.0)
    time_to_nearest_neutral = jnp.where(active, nearest_neutral_d_raw / max_speed / time_norm, 0.0)

    roi = production / (ships + 1.0)
    roi_norm = roi / jnp.float32(2.0)
    is_high_value = (production >= 3.0).astype(jnp.float32) * active.astype(jnp.float32)

    # Precompute rank matrices O(P^2) once
    ship_smaller_pp = ships[:, None] > ships[None, :]
    prod_smaller_pp = production[:, None] > production[None, :]

    ship_rank_all = _rank_norm_from_matrix(ship_smaller_pp, active)
    prod_rank_all = _rank_norm_from_matrix(prod_smaller_pp, active)
    my_ship_rank = _rank_norm_from_matrix(ship_smaller_pp, owner_is_me)
    enemy_ship_rank = _rank_norm_from_matrix(ship_smaller_pp, owner_is_enemy)
    is_my_largest = owner_is_me & (my_ship_rank >= jnp.float32(1.0 - 1e-6))
    is_enemy_largest = owner_is_enemy & (enemy_ship_rank >= jnp.float32(1.0 - 1e-6))

    net_balance = ships + incoming_me - incoming_enemy
    net_balance_signed_log = jnp.sign(net_balance) * _ships_log_norm(jnp.abs(net_balance))
    would_lose = active & owner_is_me & (incoming_enemy > ships)

    comet_remaining = _comet_remaining_life(state)
    comet_remaining_norm = comet_remaining / jnp.float32(64.0)

    # Global variables computation
    my_planet_count = jnp.sum(owner_is_me.astype(jnp.float32))
    enemy_planet_count = jnp.sum(owner_is_enemy.astype(jnp.float32))
    neutral_planet_count = jnp.sum(owner_is_neutral.astype(jnp.float32))

    my_prod = jnp.sum(jnp.where(owner_is_me, production, 0.0))
    enemy_prod = jnp.sum(jnp.where(owner_is_enemy, production, 0.0))

    my_ships = jnp.sum(jnp.where(owner_is_me, ships, 0.0))
    enemy_ships = jnp.sum(jnp.where(owner_is_enemy, ships, 0.0))

    prod_lead = (my_prod - enemy_prod) / (my_prod + enemy_prod + 1.0)
    ship_lead = (my_ships - enemy_ships) / (my_ships + enemy_ships + 1.0)

    active_comets = jnp.sum(state.comets.active.astype(jnp.float32))

    fleet_owner = fleets[:, 1]
    fleet_ships = fleets[:, 6]
    fleet_active = fleets[:, 7] > 0.0
    f_is_me = fleet_active & (fleet_owner == player_f)
    f_is_enemy = fleet_active & (fleet_owner >= 0.0) & (fleet_owner != player_f)

    my_fleet_ships = jnp.sum(jnp.where(f_is_me, fleet_ships, 0.0))
    enemy_fleet_ships = jnp.sum(jnp.where(f_is_enemy, fleet_ships, 0.0))
    my_fleet_count = jnp.sum(f_is_me.astype(jnp.float32))
    enemy_fleet_count = jnp.sum(f_is_enemy.astype(jnp.float32))

    turn = state.step.astype(jnp.float32) / jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)

    my_largest_ships = jnp.max(jnp.where(owner_is_me, ships, 0.0))
    enemy_largest_ships = jnp.max(jnp.where(owner_is_enemy, ships, 0.0))
    my_largest_prod = jnp.max(jnp.where(owner_is_me, production, 0.0))
    enemy_largest_prod = jnp.max(jnp.where(owner_is_enemy, production, 0.0))

    cur_step = state.step.astype(jnp.float32)
    ep = jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)
    spawn_arr = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.float32)
    delta = spawn_arr - cur_step
    delta = jnp.where(delta > 0.0, delta, ep)
    next_comet_in = jnp.min(delta) / ep

    is_late_game = (turn > jnp.float32(0.5)).astype(jnp.float32)

    # Expert Predictive Features: Projected Ships at k=10 and k=20
    # Note: we use do-nothing projection (baseline)
    # Using small horizon to keep it fast
    proj = project_garrison(state, horizon=20)
    ships_10 = proj.ships[:, 10]
    ships_20 = proj.ships[:, 20]
    
    # 'Slack' feature: how many ships are 'liquid' (safe to move)
    # slack = current_ships - safe_drain
    # safe_drain is defined as min(ships[t] where owned)
    from .producer import safe_drain
    s_drain = safe_drain(proj, jnp.arange(MAX_PLANETS), player_id=player)
    slack = jnp.maximum(0.0, ships - s_drain)

    planet_features = jnp.stack(
        [
            active.astype(jnp.float32),                              # 0
            owner_is_me.astype(jnp.float32),                         # 1
            owner_is_enemy.astype(jnp.float32),                      # 2
            owner_is_neutral.astype(jnp.float32),                    # 3
            _ships_log_norm(ships),                                  # 4
            production / PRODUCTION_DENOM,                           # 5
            radius / RADIUS_DENOM,                                   # 6
            x / BOARD_SIZE,                                          # 7
            y / BOARD_SIZE,                                          # 8
            dx_c,                                                    # 9
            dy_c,                                                    # 10
            dist_c,                                                  # 11
            is_orbiting.astype(jnp.float32),                         # 12
            is_comet.astype(jnp.float32),                            # 13
            orbit_r_norm,                                            # 14
            orbit_angle_sin,                                         # 15
            orbit_angle_cos,                                         # 16
            _ships_log_norm(incoming_me),                            # 17
            _ships_log_norm(incoming_enemy),                         # 18
            eta_me_norm,                                             # 19
            eta_enemy_norm,                                          # 20
            roi_norm,                                                # 21
            is_high_value,                                           # 22
            nearest_enemy_d - nearest_friend_d,                      # 23
            ship_rank_all,                                           # 24
            prod_rank_all,                                           # 25
            net_balance_signed_log,                                  # 26
            would_lose.astype(jnp.float32),                          # 27
            time_to_nearest_enemy,                                   # 28
            time_to_nearest_neutral,                                 # 29
            is_my_largest.astype(jnp.float32),                       # 30
            is_enemy_largest.astype(jnp.float32),                    # 31
            comet_remaining_norm,                                    # 32
            jnp.broadcast_to(turn, (MAX_PLANETS,)),                                            # 33
            jnp.broadcast_to(1.0 - turn, (MAX_PLANETS,)),                                      # 34
            jnp.broadcast_to(my_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)),      # 35
            jnp.broadcast_to(enemy_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)),   # 36
            jnp.broadcast_to(neutral_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)), # 37
            jnp.broadcast_to(my_prod / jnp.float32(50.0), (MAX_PLANETS,)),                     # 38
            jnp.broadcast_to(enemy_prod / jnp.float32(50.0), (MAX_PLANETS,)),                  # 39
            jnp.broadcast_to(_ships_log_norm(my_ships), (MAX_PLANETS,)),                       # 40
            jnp.broadcast_to(_ships_log_norm(enemy_ships), (MAX_PLANETS,)),                    # 41
            jnp.broadcast_to(prod_lead, (MAX_PLANETS,)),                                       # 42
            jnp.broadcast_to(ship_lead, (MAX_PLANETS,)),                                       # 43
            jnp.broadcast_to(active_comets / jnp.float32(MAX_COMET_GROUPS), (MAX_PLANETS,)),   # 44
            jnp.broadcast_to(_fleet_ships_log_norm(my_fleet_ships), (MAX_PLANETS,)),           # 45
            jnp.broadcast_to(_fleet_ships_log_norm(enemy_fleet_ships), (MAX_PLANETS,)),        # 46
            jnp.broadcast_to(my_fleet_count / jnp.float32(MAX_FLEETS), (MAX_PLANETS,)),        # 47
            jnp.broadcast_to(enemy_fleet_count / jnp.float32(MAX_FLEETS), (MAX_PLANETS,)),     # 48
            jnp.broadcast_to(_ships_log_norm(my_largest_ships), (MAX_PLANETS,)),               # 49
            jnp.broadcast_to(_ships_log_norm(enemy_largest_ships), (MAX_PLANETS,)),            # 50
            jnp.broadcast_to(my_largest_prod / PRODUCTION_DENOM, (MAX_PLANETS,)),              # 51
            jnp.broadcast_to(enemy_largest_prod / PRODUCTION_DENOM, (MAX_PLANETS,)),           # 52
            jnp.broadcast_to(next_comet_in, (MAX_PLANETS,)),                                   # 53
            jnp.broadcast_to(is_late_game, (MAX_PLANETS,)),                                    # 54
            _ships_log_norm(ships_10),                                                        # 55
            _ships_log_norm(ships_20),                                                        # 56
            _ships_log_norm(slack),                                                           # 57
        ],
        axis=-1,
    )
    planet_features = jnp.where(active[:, None], planet_features, 0.0)
    planet_mask = active

    return {
        "planet_features": planet_features,
        "planet_mask": planet_mask,
        "incoming_me": incoming_me,
        "incoming_enemy": incoming_enemy,
    }


# Pre-jitted single + batched variants for convenience.
encode_observation_jit = jax.jit(encode_observation, static_argnames=())


def encode_batch(states: OrbitWarsState, players: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Vmapped encoder. `players` has shape (B,) int32."""
    return jax.vmap(encode_observation, in_axes=(0, 0))(states, players)


encode_batch_jit = jax.jit(encode_batch)
