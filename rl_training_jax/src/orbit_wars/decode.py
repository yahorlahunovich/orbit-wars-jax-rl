"""Action decoding and grid composition for Orbit Wars.
Splits composition into two phases (Target Phase and Bucket Phase) to avoid
expensive O(P*P*B) calculations and huge intermediate tensors.
"""

from __future__ import annotations

import functools
import jax
import jax.numpy as jnp

from .constants import (
    MIN_LAUNCH_SHIPS,
    PATH_PLANET_MARGIN,
    SUN_PATH_MARGIN,
    INTERCEPT_ITERATIONS,
    BUCKET_COUNT,
)
from .geometry import (
    is_orbiting_planet,
    point_to_segment_distance,
    estimate_intercept_angles,
)
from .state import OrbitWarsState


def ship_counts_for_buckets(
    src_ships: jnp.ndarray, # (...)
    tgt_ships: jnp.ndarray, # (...)
    inc_me: jnp.ndarray,    # (...)
    inc_en: jnp.ndarray,    # (...)
) -> jnp.ndarray:
    """Return an array of ship counts for each of the 8 buckets."""
    b0 = src_ships * 0.10
    b1 = src_ships * 0.25
    b2 = src_ships * 0.33
    b3 = src_ships * 0.50
    b4 = src_ships * 1.00
    
    needed = jnp.maximum(0.0, tgt_ships + inc_en - inc_me)
    b5 = needed + 1.0
    b6 = needed + 5.0
    b7 = needed + 10.0
    
    bs = jnp.broadcast_arrays(b0, b1, b2, b3, b4, b5, b6, b7)
    counts = jnp.stack(bs, axis=-1)
    return jnp.clip(jnp.floor(counts), 1.0, jnp.maximum(1.0, src_ships[..., None]))


def bucket_validity_mask(
    ship_counts: jnp.ndarray, source_ships: jnp.ndarray
) -> jnp.ndarray:
    src = source_ships[..., None]
    has_enough_to_launch = src >= jnp.float32(MIN_LAUNCH_SHIPS)
    return has_enough_to_launch & (ship_counts > 0.0) & (ship_counts <= src)


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
    ox = planet_x
    oy = planet_y
    obs_r = planet_radius + margin
    is_obstacle = planet_active

    for _ in range(start_x.ndim):
        ox = ox[None, ...]
        oy = oy[None, ...]
        obs_r = obs_r[None, ...]
        is_obstacle = is_obstacle[None, ...]

    sx = start_x[..., None]
    sy = start_y[..., None]
    tx = target_x[..., None]
    ty = target_y[..., None]

    x_min = jnp.minimum(sx, tx) - obs_r
    x_max = jnp.maximum(sx, tx) + obs_r
    y_min = jnp.minimum(sy, ty) - obs_r
    y_max = jnp.maximum(sy, ty) + obs_r
    
    in_box = (ox >= x_min) & (ox <= x_max) & (oy >= y_min) & (oy <= y_max)
    d = point_to_segment_distance(ox, oy, sx, sy, tx, ty)
    
    # Ignore obstacles at the exact start or end point (self)
    d_start = jnp.sqrt((ox - sx)**2 + (oy - sy)**2)
    d_target = jnp.sqrt((ox - tx)**2 + (oy - ty)**2)
    is_not_self = (d_start > 1e-3) & (d_target > 1e-3)
    
    return jnp.any((d <= obs_r) & is_obstacle & is_not_self, axis=-1)


def compose_target_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
    incoming_me: jnp.ndarray,
    incoming_enemy: jnp.ndarray,
    *,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    sun_path_margin: float = SUN_PATH_MARGIN,
    path_planet_margin: float = PATH_PLANET_MARGIN,
    enable_planet_block: bool = True,
) -> dict[str, jnp.ndarray]:
    """Phase 1: Compute which (source, target) pairs are potentially valid."""
    planets = state.planets
    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    player_f = jnp.float32(player)
    source_valid = active & (owner == player_f)
    target_valid = active

    x, y, radius, ships, pids = planets[:, 2], planets[:, 3], planets[:, 4], planets[:, 5], planets[:, 0].astype(jnp.int32)

    tgt_orbiting = is_orbiting_planet(x, y, radius)
    # Representative ships (P,)
    rep_ships = jnp.maximum(1.0, jnp.floor(ships * 0.5))
    
    from .geometry import precompute_comet_trajectories
    is_comet, trajectories, valid_time = precompute_comet_trajectories(
        state.comets.active, state.comets.planet_ids, state.comets.path_index,
        state.comets.paths, state.comets.path_lengths, pids
    )
    
    def _row_intercept(sx, sy, sr):
        # We want to check all targets (P,) for this one source.
        # Outputs (P,) tensors.
        return estimate_intercept_angles(
            sx, sy, sr, x, y, radius, tgt_orbiting, is_comet, 
            trajectories, valid_time, rep_ships,
            state.angular_velocity, state.ship_speed,
            n_iter=intercept_iterations, sun_margin=sun_path_margin,
        )

    angle, aim_x, aim_y, sun_blocks = jax.vmap(_row_intercept)(x, y, radius)

    if enable_planet_block:
        planet_blocks = path_blocked_by_planets(
            x[:, None], y[:, None], x[None, :], y[None, :], x, y, radius, active, margin=path_planet_margin,
        )
    else:
        planet_blocks = jnp.zeros((planets.shape[0], planets.shape[0]), dtype=jnp.bool_)

    pair_valid = source_valid[:, None] & target_valid[None, :]
    has_enough = (ships >= jnp.float32(MIN_LAUNCH_SHIPS))
    target_mask = pair_valid & has_enough[:, None] & (~sun_blocks) & (~planet_blocks)

    return {
        "source_valid_any": jnp.any(target_mask, axis=-1),
        "target_mask": target_mask,
        "from_ids": pids,
        "is_comet": is_comet,
        "trajectories": trajectories,
        "valid_time": valid_time,
        "incoming_me": incoming_me,
        "incoming_enemy": incoming_enemy,
    }


def compose_bucket_grid(
    state: OrbitWarsState,
    target_idx: jnp.ndarray, # (P,) chosen target per source
    phase1_results: dict,
    *,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    sun_path_margin: float = SUN_PATH_MARGIN,
    **_kwargs,
) -> dict[str, jnp.ndarray]:
    """Phase 2: Compute exact angles and bucket validity for CHOSEN targets."""
    planets = state.planets
    x, y, radius, ships = planets[:, 2], planets[:, 3], planets[:, 4], planets[:, 5]
    
    tx = x[target_idx]
    ty = y[target_idx]
    tr = radius[target_idx]
    torb = is_orbiting_planet(tx, ty, tr)
    tcom = phase1_results["is_comet"][target_idx]
    ttraj = phase1_results["trajectories"][target_idx]
    tvt = phase1_results["valid_time"][target_idx]
    
    tgt_ships = ships[target_idx]
    inc_me = phase1_results["incoming_me"][target_idx]
    inc_en = phase1_results["incoming_enemy"][target_idx]
    
    # ship_counts: (P, B)
    ship_counts = ship_counts_for_buckets(ships, tgt_ships, inc_me, inc_en)
    
    def _bucket_intercept(sx, sy, sr, tx, ty, tr, torb, tcom, ttraj, tvt, sc):
        # All inputs are scalars except sc which is (B,)
        # Returns (B,) tensors.
        return estimate_intercept_angles(
            sx, sy, sr, tx, ty, tr, torb, tcom, ttraj, tvt, sc,
            state.angular_velocity, state.ship_speed,
            n_iter=intercept_iterations, sun_margin=sun_path_margin,
        )

    angle, aim_x, aim_y, sun_blocks = jax.vmap(_bucket_intercept)(
        x, y, radius, tx, ty, tr, torb, tcom, ttraj, tvt, ship_counts
    )
    
    bucket_valid = bucket_validity_mask(ship_counts, ships) & (~sun_blocks)
    
    return {
        "angle": angle,
        "ship_counts": ship_counts,
        "bucket_valid": bucket_valid,
    }


def compose_full_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
    incoming_me: jnp.ndarray | None = None,
    incoming_enemy: jnp.ndarray | None = None,
    *,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    sun_path_margin: float = SUN_PATH_MARGIN,
    path_planet_margin: float = PATH_PLANET_MARGIN,
    enable_planet_block: bool = True,
) -> dict[str, jnp.ndarray]:
    """Compatibility: Builds the full (P, P, B) grid. SLOW."""
    planets = state.planets
    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    player_f = jnp.float32(player)
    source_valid = active & (owner == player_f)
    target_valid = active
    x, y, radius, ships, pids = planets[:, 2], planets[:, 3], planets[:, 4], planets[:, 5], planets[:, 0].astype(jnp.int32)

    if incoming_me is None or incoming_enemy is None:
        from .features_jax import _fleet_projections
        incoming_me, incoming_enemy, _, _ = _fleet_projections(state, player_f)

    # (P, P, B)
    ship_counts = ship_counts_for_buckets(ships[:, None], ships[None, :], incoming_me[None, :], incoming_enemy[None, :])

    from .geometry import precompute_comet_trajectories
    is_comet, trajectories, valid_time = precompute_comet_trajectories(
        state.comets.active, state.comets.planet_ids, state.comets.path_index,
        state.comets.paths, state.comets.path_lengths, pids
    )
    tgt_orbiting = is_orbiting_planet(x, y, radius)

    def _full_intercept(sx, sy, sr, sc_row):
        # sx, sy, sr are scalars. sc_row is (P, B).
        # We want to check all targets (P,) with their buckets (B,).
        # Returns (P, B) tensors.
        return estimate_intercept_angles(
            sx, sy, sr, x, y, radius, tgt_orbiting, is_comet,
            trajectories, valid_time, sc_row,
            state.angular_velocity, state.ship_speed,
            n_iter=intercept_iterations, sun_margin=sun_path_margin,
        )

    angle, aim_x, aim_y, sun_blocks = jax.vmap(_full_intercept)(x, y, radius, ship_counts)

    if enable_planet_block:
        planet_blocks = path_blocked_by_planets(x[:, None], y[:, None], x[None, :], y[None, :], x, y, radius, active, margin=path_planet_margin)
        planet_blocks = planet_blocks[:, :, None]
    else:
        planet_blocks = jnp.zeros((planets.shape[0], planets.shape[0], 1), dtype=jnp.bool_)

    pair_valid = source_valid[:, None] & target_valid[None, :]
    bucket_valid = bucket_validity_mask(ship_counts, ships)
    full_valid = pair_valid[..., None] & bucket_valid & (~sun_blocks) & (~planet_blocks)

    return {
        "source_valid": source_valid,
        "target_valid": target_valid,
        "pair_valid": pair_valid,
        "bucket_valid": bucket_valid,
        "sun_blocks": sun_blocks,
        "planet_blocks": planet_blocks,
        "full_valid": full_valid,
        "from_ids": pids,
        "angle": angle,
        "ship_counts": ship_counts,
    }

def compose_action_grid(state, player, **kwargs):
    return compose_full_grid(state, player, **kwargs)

def pack_action_row(from_id, angle, ships, valid):
    row = jnp.stack([from_id.astype(jnp.float32), angle.astype(jnp.float32), jnp.floor(ships).astype(jnp.float32)], axis=-1)
    valid_f = valid.astype(jnp.float32)
    return row * valid_f[..., None], valid_f

def launch_angle(src_x, src_y, tgt_x, tgt_y):
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)

def path_crosses_sun(src_x, src_y, tgt_x, tgt_y, margin=SUN_PATH_MARGIN):
    from .geometry import sun_hit
    return sun_hit(src_x, src_y, tgt_x, tgt_y, margin=margin)
