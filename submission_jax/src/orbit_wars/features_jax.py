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
    DEFAULT_SHIP_SPEED,
    MAX_COMET_GROUPS,
    MAX_FLEETS,
    MAX_PLANETS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .geometry import fleet_speed
from .state import OrbitWarsState

PLANET_FEATURE_DIM = 31
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


def _fleet_incoming_to_planet(state: OrbitWarsState) -> jnp.ndarray:
    """Return (F, P) bool matrix: True if fleet f is heading toward planet p.

    Heuristic: project (planet - fleet) onto the fleet's heading direction.
    Considered "incoming" iff the projection is positive (fleet is moving
    toward the planet) AND the perpendicular distance is within the planet's
    radius + a small tolerance (so we count fleets on a near-collision path).
    """
    planets = state.planets
    fleets = state.fleets

    fx = fleets[:, 2]
    fy = fleets[:, 3]
    angle = fleets[:, 4]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    fleet_active = fleets[:, 7] > 0.0

    px = planets[:, 2]
    py = planets[:, 3]
    radius = planets[:, 4]
    planet_active = planets[:, 7] > 0.0

    dx = px[None, :] - fx[:, None]     # (F, P)
    dy = py[None, :] - fy[:, None]
    proj = dx * cos_a[:, None] + dy * sin_a[:, None]            # along heading
    perp = jnp.abs(-dx * sin_a[:, None] + dy * cos_a[:, None])  # perpendicular
    near_path = (perp < (radius[None, :] + INCOMING_PERP_TOL))
    heading_toward = proj > 0.0

    return heading_toward & near_path & fleet_active[:, None] & planet_active[None, :]


def _nearest_distance_to_subset(
    planets: jnp.ndarray,
    subset_mask: jnp.ndarray,
) -> jnp.ndarray:
    """For each planet, distance to the nearest other planet in `subset_mask`.

    Returns DIST_DENOM if the subset is empty.
    """
    x = planets[:, 2]
    y = planets[:, 3]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy)
    big = jnp.float32(1e6)
    # Exclude self and rows outside the subset.
    self_mask = jnp.eye(MAX_PLANETS, dtype=jnp.bool_)
    masked = jnp.where(subset_mask[None, :] & (~self_mask), dist, big)
    nearest = jnp.min(masked, axis=-1)
    # Convert "no subset member" sentinel (big) to a finite max distance.
    nearest = jnp.where(nearest >= big, DIST_DENOM * 3.0, nearest)
    return nearest


def _rank_norm(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """For each entry, fraction of *masked* entries with a strictly smaller value.

    For inactive entries (`~mask`) returns 0. Output in `[0, 1]`. 1.0 means
    "largest in the masked set"; 0.0 means "smallest in the masked set".
    """
    eligible_count = jnp.sum(mask.astype(jnp.float32))
    smaller = jnp.sum(
        (values[:, None] > values[None, :])
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
        fleet_features  (F, F_FLEET)   float32
        fleet_mask      (F,)           bool
        global_features (F_GLOBAL,)    float32

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

    # Per-planet incoming ships (split by my / enemy fleets).
    inc_fp = _fleet_incoming_to_planet(state)                    # (F, P) bool
    fleet_owner = fleets[:, 1]
    fleet_ships = fleets[:, 6]
    fleet_active = fleets[:, 7] > 0.0
    f_is_me = fleet_active & (fleet_owner == player_f)
    f_is_enemy = fleet_active & (fleet_owner >= 0.0) & (fleet_owner != player_f)
    incoming_me = jnp.sum(jnp.where(inc_fp & f_is_me[:, None], fleet_ships[:, None], 0.0), axis=0)
    incoming_enemy = jnp.sum(jnp.where(inc_fp & f_is_enemy[:, None], fleet_ships[:, None], 0.0), axis=0)

    nearest_enemy_d_raw = _nearest_distance_to_subset(planets, owner_is_enemy)
    nearest_friend_d_raw = _nearest_distance_to_subset(planets, owner_is_me)
    nearest_neutral_d_raw = _nearest_distance_to_subset(planets, owner_is_neutral)
    nearest_enemy_d = nearest_enemy_d_raw / DIST_DENOM
    nearest_friend_d = nearest_friend_d_raw / DIST_DENOM

    max_speed = state.ship_speed
    time_norm = jnp.float32(100.0)
    time_to_nearest_enemy = jnp.where(active, nearest_enemy_d_raw / max_speed / time_norm, 0.0)
    time_to_nearest_neutral = jnp.where(active, nearest_neutral_d_raw / max_speed / time_norm, 0.0)

    roi = production / (ships + 1.0)
    roi_norm = roi / jnp.float32(2.0)
    is_high_value = (production >= 3.0).astype(jnp.float32) * active.astype(jnp.float32)

    # Rankings among active planets.
    ship_rank_all = _rank_norm(ships, active)
    prod_rank_all = _rank_norm(production, active)
    # "Largest" flags (rank == 1.0 within the owner group, i.e. strictly largest).
    my_ship_rank = _rank_norm(ships, owner_is_me)
    enemy_ship_rank = _rank_norm(ships, owner_is_enemy)
    is_my_largest = owner_is_me & (my_ship_rank >= jnp.float32(1.0 - 1e-6))
    is_enemy_largest = owner_is_enemy & (enemy_ship_rank >= jnp.float32(1.0 - 1e-6))

    net_balance = ships + incoming_me - incoming_enemy
    net_balance_signed_log = jnp.sign(net_balance) * _ships_log_norm(jnp.abs(net_balance))
    would_lose = active & owner_is_me & (incoming_enemy > ships)

    comet_remaining = _comet_remaining_life(state)
    # MAX_COMET_PATH_LEN = 64 → divide by 64 for a soft [0, 1].
    comet_remaining_norm = comet_remaining / jnp.float32(64.0)

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
            roi_norm,                                                # 19
            is_high_value,                                           # 20
            nearest_enemy_d - nearest_friend_d,                      # 21
            ship_rank_all,                                           # 22
            prod_rank_all,                                           # 23
            net_balance_signed_log,                                  # 24
            would_lose.astype(jnp.float32),                          # 25
            time_to_nearest_enemy,                                   # 26
            time_to_nearest_neutral,                                 # 27
            is_my_largest.astype(jnp.float32),                       # 28
            is_enemy_largest.astype(jnp.float32),                    # 29
            comet_remaining_norm,                                    # 30
        ],
        axis=-1,
    )                                                                # (P, 31)
    planet_features = jnp.where(active[:, None], planet_features, 0.0)
    planet_mask = active

    # ---------------- Fleet features --------------------------------------
    fx = fleets[:, 2]
    fy = fleets[:, 3]
    angle = fleets[:, 4]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    speed = fleet_speed(fleet_ships, state.ship_speed)
    vx = cos_a * speed
    vy = sin_a * speed

    # Position relative to the sun. Bearing-radial frame: rotate heading into
    # the local frame where +x points outward from the sun. Useful for
    # distinguishing inbound-to-sun (suicidal) fleets from outbound ones.
    fdx_c = fx - CENTER
    fdy_c = fy - CENTER
    fdist_c = jnp.sqrt(fdx_c * fdx_c + fdy_c * fdy_c)
    safe_dist = jnp.maximum(fdist_c, 1e-3)
    radial_cos = fdx_c / safe_dist           # unit outward direction
    radial_sin = fdy_c / safe_dist
    # bearing_radial = heading rotated by -radial_angle, i.e. project onto outward / tangent.
    bearing_radial_cos = cos_a * radial_cos + sin_a * radial_sin    # +1 = outbound, -1 = inbound
    bearing_radial_sin = -cos_a * radial_sin + sin_a * radial_cos   # tangential component
    is_heading_inward = (bearing_radial_cos < 0.0)                  # bool

    fleet_features = jnp.stack(
        [
            fleet_active.astype(jnp.float32),                        # 0
            f_is_me.astype(jnp.float32),                             # 1
            f_is_enemy.astype(jnp.float32),                          # 2
            _fleet_ships_log_norm(fleet_ships),                      # 3
            cos_a,                                                   # 4
            sin_a,                                                   # 5
            fx / BOARD_SIZE,                                         # 6
            fy / BOARD_SIZE,                                         # 7
            speed / state.ship_speed,                                # 8
            vx / state.ship_speed,                                   # 9
            vy / state.ship_speed,                                   # 10
            fdist_c / DIST_DENOM,                                    # 11
            bearing_radial_cos,                                      # 12  (outbound/inbound)
            bearing_radial_sin,                                      # 13  (tangential)
            is_heading_inward.astype(jnp.float32),                   # 14
        ],
        axis=-1,
    )                                                                # (F, 15)
    fleet_features = jnp.where(fleet_active[:, None], fleet_features, 0.0)
    fleet_mask = fleet_active

    # ---------------- Global features --------------------------------------
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

    my_fleet_ships = jnp.sum(jnp.where(f_is_me, fleet_ships, 0.0))
    enemy_fleet_ships = jnp.sum(jnp.where(f_is_enemy, fleet_ships, 0.0))
    my_fleet_count = jnp.sum(f_is_me.astype(jnp.float32))
    enemy_fleet_count = jnp.sum(f_is_enemy.astype(jnp.float32))

    turn = state.step.astype(jnp.float32) / jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)

    my_largest_ships = jnp.max(jnp.where(owner_is_me, ships, 0.0))
    enemy_largest_ships = jnp.max(jnp.where(owner_is_enemy, ships, 0.0))
    my_largest_prod = jnp.max(jnp.where(owner_is_me, production, 0.0))
    enemy_largest_prod = jnp.max(jnp.where(owner_is_enemy, production, 0.0))

    # Steps until next comet spawn (normalized to episode_steps).
    cur_step = state.step.astype(jnp.float32)
    ep = jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)
    spawn_arr = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.float32)
    delta = spawn_arr - cur_step
    # Replace past spawn steps with a large positive so they don't win the min.
    delta = jnp.where(delta > 0.0, delta, ep)
    next_comet_in = jnp.min(delta) / ep                    # in (0, 1]

    is_late_game = (turn > jnp.float32(0.5)).astype(jnp.float32)

    global_features = jnp.stack(
        [
            turn,                                                    # 0
            1.0 - turn,                                              # 1
            my_planet_count / jnp.float32(MAX_PLANETS),              # 2
            enemy_planet_count / jnp.float32(MAX_PLANETS),           # 3
            neutral_planet_count / jnp.float32(MAX_PLANETS),         # 4
            my_prod / jnp.float32(50.0),                             # 5
            enemy_prod / jnp.float32(50.0),                          # 6
            _ships_log_norm(my_ships),                               # 7
            _ships_log_norm(enemy_ships),                            # 8
            prod_lead,                                               # 9
            ship_lead,                                               # 10
            active_comets / jnp.float32(MAX_COMET_GROUPS),           # 11
            _fleet_ships_log_norm(my_fleet_ships),                   # 12
            _fleet_ships_log_norm(enemy_fleet_ships),                # 13
            my_fleet_count / jnp.float32(MAX_FLEETS),                # 14
            enemy_fleet_count / jnp.float32(MAX_FLEETS),             # 15
            _ships_log_norm(my_largest_ships),                       # 16
            _ships_log_norm(enemy_largest_ships),                    # 17
            my_largest_prod / PRODUCTION_DENOM,                      # 18
            enemy_largest_prod / PRODUCTION_DENOM,                   # 19
            next_comet_in,                                           # 20
            is_late_game,                                            # 21
        ],
        axis=-1,
    )                                                                # (22,)

    return {
        "planet_features": planet_features,
        "planet_mask": planet_mask,
    }


# Pre-jitted single + batched variants for convenience.
encode_observation_jit = jax.jit(encode_observation, static_argnames=())


def encode_batch(states: OrbitWarsState, players: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Vmapped encoder. `players` has shape (B,) int32."""
    return jax.vmap(encode_observation, in_axes=(0, 0))(states, players)


encode_batch_jit = jax.jit(encode_batch)
