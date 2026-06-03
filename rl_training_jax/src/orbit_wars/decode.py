"""Pure-JAX geometry decoder for Orbit Wars actions.
Updated to match the 1100 ELO heuristic notebook's trajectory logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import CENTER, SUN_RADIUS
from .geometry import (
    estimate_intercept_angles,
    is_orbiting_planet,
    point_to_segment_distance,
    safe_angle,
    sun_hit,
)
from .state import OrbitWarsState

BUCKET_COUNT = 8
MIN_LAUNCH_SHIPS = 4
SUN_PATH_MARGIN = 1.5
PATH_PLANET_MARGIN = 1.0
INTERCEPT_ITERATIONS = 6  # Match notebook


# Launch offset to avoid spawning fleets inside the source planet.
LAUNCH_OFFSET_PADDING = 0.1


def ship_counts_for_buckets(
    source_ships: jnp.ndarray, target_ships: jnp.ndarray, incoming_me: jnp.ndarray, incoming_enemy: jnp.ndarray
) -> jnp.ndarray:
    """Return integer-valued ship counts for every bucket index."""
    src = source_ships[..., None]
    tgt = target_ships[..., None]
    inc_me = incoming_me[..., None]
    inc_en = incoming_enemy[..., None]
    
    b0 = src * 0.25
    b1 = src * 0.50
    b2 = src * 0.75
    b3 = src * 1.00
    b4 = tgt + 1.0
    b5 = tgt + src * 0.50
    b6 = jnp.maximum(0.0, tgt + inc_en - inc_me) + 1.0
    b7 = jnp.maximum(0.0, tgt + inc_en - inc_me) + src * 0.25
    
    b0, b1, b2, b3, b4, b5, b6, b7 = jnp.broadcast_arrays(b0, b1, b2, b3, b4, b5, b6, b7)
    raw = jnp.concatenate([b0, b1, b2, b3, b4, b5, b6, b7], axis=-1)
    raw = jnp.maximum(raw, jnp.float32(MIN_LAUNCH_SHIPS))
    return jnp.floor(raw)


def bucket_validity_mask(
    ship_counts: jnp.ndarray, source_ships: jnp.ndarray
) -> jnp.ndarray:
    src = source_ships[..., None]
    return (ship_counts > 0.0) & (ship_counts <= src)


def launch_angle(
    src_x: jnp.ndarray, src_y: jnp.ndarray,
    tgt_x: jnp.ndarray, tgt_y: jnp.ndarray,
    margin: float = SUN_PATH_MARGIN,
) -> jnp.ndarray:
    return safe_angle(src_x, src_y, tgt_x, tgt_y, sun_margin=margin)


def path_crosses_sun(
    src_x: jnp.ndarray, src_y: jnp.ndarray,
    tgt_x: jnp.ndarray, tgt_y: jnp.ndarray,
    margin: float = 0.0,
) -> jnp.ndarray:
    return sun_hit(src_x, src_y, tgt_x, tgt_y, margin=margin)


def path_blocked_by_planets(
    start_x: jnp.ndarray,
    start_y: jnp.ndarray,
    target_x: jnp.ndarray,
    target_y: jnp.ndarray,
    planet_x: jnp.ndarray,
    planet_y: jnp.ndarray,
    planet_radius: jnp.ndarray,
    planet_active: jnp.ndarray,
    margin: float = PATH_PLANET_MARGIN,
) -> jnp.ndarray:
    p = planet_x.shape[0]
    slot = jnp.arange(p)
    src_i = slot[:, None, None]
    tgt_i = slot[None, :, None]
    obs_i = slot[None, None, :]

    is_obstacle = (
        planet_active[None, None, :]
        & (obs_i != src_i)
        & (obs_i != tgt_i)
    )
    obs_r = (planet_radius + margin)[None, None, :]
    ox = planet_x[None, None, :]
    oy = planet_y[None, None, :]

    sx = start_x[:, :, None]
    sy = start_y[:, :, None]
    tx = target_x[:, :, None]
    ty = target_y[:, :, None]

    d = point_to_segment_distance(ox, oy, sx, sy, tx, ty)
    return jnp.any((d <= obs_r) & is_obstacle, axis=2)


def compose_action_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
    *,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    sun_path_margin: float = SUN_PATH_MARGIN,
    path_planet_margin: float = PATH_PLANET_MARGIN,
    enable_planet_block: bool = True,
    enable_incoming_projection: bool = True,
) -> dict[str, jnp.ndarray]:
    planets = state.planets
    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    player_f = jnp.float32(player)
    source_valid = active & (owner == player_f)
    target_valid = active

    x = planets[:, 2]
    y = planets[:, 3]
    radius = planets[:, 4]
    ships = planets[:, 5]
    pids = planets[:, 0].astype(jnp.int32)

    tgt_orbiting = is_orbiting_planet(x, y, radius)

    if enable_incoming_projection:
        from .features_jax import _fleet_projections
        incoming_me, incoming_enemy, _, _ = _fleet_projections(state, player_f)
    else:
        incoming_me = jnp.zeros_like(ships)
        incoming_enemy = jnp.zeros_like(ships)

    src_ships_grid = ships[:, None]
    tgt_ships_grid = ships[None, :]
    inc_me_grid = incoming_me[None, :]
    inc_en_grid = incoming_enemy[None, :]
    ship_counts = ship_counts_for_buckets(src_ships_grid, tgt_ships_grid, inc_me_grid, inc_en_grid)

    p_count = planets.shape[0]
    bucket_axis = ship_counts.shape[-1]
    
    src_x_b = x[:, None, None]
    src_y_b = y[:, None, None]
    src_r_b = radius[:, None, None]
    
    tgt_x_b = x[None, :, None]
    tgt_y_b = y[None, :, None]
    tgt_r_b = radius[None, :, None]
    tgt_orb_b = tgt_orbiting[None, :, None]

    # Precompute comet trajectories to avoid doing it inside the geometry loops
    from .geometry import precompute_comet_trajectories
    is_comet, trajectories, valid_time = precompute_comet_trajectories(
        state.comets.active, state.comets.planet_ids, state.comets.path_index,
        state.comets.paths, state.comets.path_lengths, pids
    )
    
    tgt_com_b = is_comet[None, :, None]
    tgt_traj_b = trajectories[None, :, None, :, :]  # (1, P, 1, L, 2)
    tgt_vt_b = valid_time[None, :, None, :]  # (1, P, 1, L)

    # Perform intercept estimation (iterative)
    # JAX will automatically broadcast (P, 1, 1), (1, P, 1), and (P, P, B) arrays internally.
    angle, aim_x, aim_y, sun_blocks = estimate_intercept_angles(
        src_x_b, src_y_b, src_r_b,
        tgt_x_b, tgt_y_b, tgt_r_b,
        tgt_orb_b, tgt_com_b, tgt_traj_b, tgt_vt_b,
        ship_counts,
        state.angular_velocity, state.ship_speed,
        n_iter=intercept_iterations,
        sun_margin=sun_path_margin,
    )

    if enable_planet_block:
        center_x_2d = jnp.broadcast_to(x[:, None], (p_count, p_count))
        center_y_2d = jnp.broadcast_to(y[:, None], (p_count, p_count))
        tgt_x_2d = jnp.broadcast_to(x[None, :], (p_count, p_count))
        tgt_y_2d = jnp.broadcast_to(y[None, :], (p_count, p_count))
        pb_2d = path_blocked_by_planets(
            center_x_2d, center_y_2d, tgt_x_2d, tgt_y_2d, x, y, radius, active, margin=path_planet_margin,
        )
        planet_blocks = jnp.broadcast_to(pb_2d[:, :, None], (p_count, p_count, bucket_axis))
    else:
        planet_blocks = jnp.zeros_like(sun_blocks, dtype=jnp.bool_)

    self_target = jnp.eye(p_count, dtype=jnp.bool_)
    pair_valid = source_valid[:, None] & target_valid[None, :]

    bucket_valid = bucket_validity_mask(ship_counts, src_ships_grid)
    full_valid = pair_valid[..., None] & bucket_valid & (~sun_blocks) & (~planet_blocks)

    from_ids = planets[:, 0]

    return {
        "source_valid": source_valid,
        "target_valid": target_valid,
        "angle": angle,
        "aim_x": aim_x,
        "aim_y": aim_y,
        "sun_blocks": sun_blocks,
        "planet_blocks": planet_blocks,
        "self_target": self_target,
        "target_valid_pair": target_valid[None, :],
        "ship_counts": ship_counts,
        "bucket_valid": bucket_valid,
        "pair_valid": pair_valid,
        "full_valid": full_valid,
        "from_ids": from_ids,
    }


def pack_action_row(
    from_id: jnp.ndarray,
    angle: jnp.ndarray,
    ships: jnp.ndarray,
    valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    row = jnp.stack([
        from_id.astype(jnp.float32),
        angle.astype(jnp.float32),
        jnp.floor(ships).astype(jnp.float32),
    ])
    valid_f = valid.astype(jnp.float32)
    return row * valid_f, valid_f
