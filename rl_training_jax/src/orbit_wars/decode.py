"""Pure-JAX geometry decoder for Orbit Wars actions.

Given a chosen `(source_planet, target_planet, bucket)` triple, produce a
legal `[from_id, angle, num_ships]` action row plus validity flags. All
operations are vectorized and vmap-friendly — Phase 4 (rollout) will broadcast
over batch × source.

Ship-bucket scheme (BUCKET_COUNT = 8):

    0  25%  of source ships    (min 4 ships)
    1  50%  of source ships
    2  75%  of source ships
    3 100%  of source ships    (all-in)
    4  target_ships + 1        (minimal capture)
    5  target_ships + 50% src  (capture with reserve)
    6  target_ships + inc_enemy - inc_allied + 1 (smart capture minimal)
    7  target_ships + inc_enemy - inc_allied + 25% src (smart capture reserve)

Buckets are masked invalid when the computed ship count is <= 0 or exceeds the
source planet's current ship count.

A move is masked invalid when:

- source planet is not active or not owned by `player`;
- target planet is not active;
- the chosen ship count is 0 or > source ships;
- the straight-line path from source to target crosses the sun;
- the source and target are the same (degenerate self-launch).
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
INTERCEPT_ITERATIONS = 5


# Launch offset to avoid spawning fleets inside the source planet.
LAUNCH_OFFSET_PADDING = 0.1


def ship_counts_for_buckets(
    source_ships: jnp.ndarray, target_ships: jnp.ndarray, incoming_me: jnp.ndarray, incoming_enemy: jnp.ndarray
) -> jnp.ndarray:
    """Return integer-valued ship counts for every bucket index.

    Inputs broadcast against each other. Output shape = broadcasted shape +
    `(BUCKET_COUNT,)`.
    """
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
    # Floor to int while keeping floats (the env stores ships as float32 ints).
    return jnp.floor(raw)


def bucket_validity_mask(
    ship_counts: jnp.ndarray, source_ships: jnp.ndarray
) -> jnp.ndarray:
    """Bool mask of buckets that can legally fire.

    `ship_counts` has the bucket dim last; `source_ships` is broadcast.
    """
    src = source_ships[..., None]
    return (ship_counts > 0.0) & (ship_counts <= src)


def path_crosses_sun(
    src_x: jnp.ndarray, src_y: jnp.ndarray,
    tgt_x: jnp.ndarray, tgt_y: jnp.ndarray,
    margin: float = 0.0,
) -> jnp.ndarray:
    """Does the straight segment src→tgt pass within SUN_RADIUS (+margin) of centre?"""
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
    """True when a third active planet intersects the start→target segment.

    Shape: inputs `(P_src, P_tgt)` for start/target; planet arrays `(P,)`.
    Returns `(P_src, P_tgt)`.
    """
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


def launch_angle(
    src_x: jnp.ndarray, src_y: jnp.ndarray,
    tgt_x: jnp.ndarray, tgt_y: jnp.ndarray,
) -> jnp.ndarray:
    """`atan2(dy, dx)` aim. Intercept correction lives in a higher layer."""
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)


def compose_action_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
    *,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    enable_planet_block: bool = True,
    enable_incoming_projection: bool = True,
) -> dict[str, jnp.ndarray]:
    """Pre-compute everything the policy/rollout needs about every
    (source, target, bucket) triple in a single state.

    Returns a dict with all (P_src, P_tgt) / (P_src, P_tgt, BUCKETS) shaped
    arrays:

        source_valid   (P,)             bool          source planet owned by player
        target_valid   (P,)             bool          target planet is active
        angle          (P, P, BUCKETS)  float32       safe intercept aim per bucket
        sun_blocks     (P, P, BUCKETS)  bool          launch→aim crosses sun
        planet_blocks  (P, P, BUCKETS)  bool          another planet blocks path
        self_target    (P, P)           bool          true on the diagonal
        target_valid_pair (P, P)        bool          target_valid AND not self
        ship_counts    (P, P, BUCKETS)  float32       per-bucket ship count to send
        bucket_valid   (P, P, BUCKETS)  bool          ship count fits source's reserve
        pair_valid     (P, P)           bool          source_valid AND target_valid_pair
        full_valid     (P, P, BUCKETS)  bool          pair & bucket & !sun & !planet block
        from_ids       (P,)             float32       planet id per source slot
    """
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

    tgt_orbiting = is_orbiting_planet(x, y, radius)  # (P,)

    if enable_incoming_projection:
        from .features_jax import _fleet_projections
        incoming_me, incoming_enemy, _, _ = _fleet_projections(state, player_f)
    else:
        incoming_me = jnp.zeros_like(ships)
        incoming_enemy = jnp.zeros_like(ships)

    src_ships_grid = ships[:, None]                  # (P, 1)
    tgt_ships_grid = ships[None, :]                  # (1, P)
    inc_me_grid = incoming_me[None, :]               # (1, P)
    inc_en_grid = incoming_enemy[None, :]            # (1, P)
    ship_counts = ship_counts_for_buckets(src_ships_grid, tgt_ships_grid, inc_me_grid, inc_en_grid)  # (P, P, B)

    p_count = planets.shape[0]
    bucket_axis = ship_counts.shape[-1]
    src_x_b = jnp.broadcast_to(x[:, None, None], (p_count, p_count, bucket_axis))
    src_y_b = jnp.broadcast_to(y[:, None, None], (p_count, p_count, bucket_axis))
    tgt_x_b = jnp.broadcast_to(x[None, :, None], (p_count, p_count, bucket_axis))
    tgt_y_b = jnp.broadcast_to(y[None, :, None], (p_count, p_count, bucket_axis))
    tgt_orb_b = jnp.broadcast_to(
        tgt_orbiting[None, :, None], (p_count, p_count, bucket_axis),
    )

    _raw_angle, aim_x, aim_y = estimate_intercept_angles(
        src_x_b, src_y_b, tgt_x_b, tgt_y_b, tgt_orb_b, ship_counts,
        state.angular_velocity, state.ship_speed, n_iter=intercept_iterations,
    )

    center_x = jnp.broadcast_to(x[:, None, None], (p_count, p_count, bucket_axis))
    center_y = jnp.broadcast_to(y[:, None, None], (p_count, p_count, bucket_axis))
    # Launch direction toward intercept; detour if the centre→aim segment crosses the sun.
    angle = safe_angle(center_x, center_y, aim_x, aim_y, sun_margin=SUN_PATH_MARGIN)

    src_radius_b = jnp.broadcast_to(radius[:, None, None], (p_count, p_count, bucket_axis))
    start_x = center_x + jnp.cos(angle) * (src_radius_b + LAUNCH_OFFSET_PADDING)
    start_y = center_y + jnp.sin(angle) * (src_radius_b + LAUNCH_OFFSET_PADDING)
    # Mask when centre→aim crosses the sun (matches heuristic pre-filter).
    sun_blocks = path_crosses_sun(center_x, center_y, aim_x, aim_y, margin=SUN_PATH_MARGIN)
    if enable_planet_block:
        center_x_2d = jnp.broadcast_to(x[:, None], (p_count, p_count))
        center_y_2d = jnp.broadcast_to(y[:, None], (p_count, p_count))
        tgt_x_2d = jnp.broadcast_to(x[None, :], (p_count, p_count))
        tgt_y_2d = jnp.broadcast_to(y[None, :], (p_count, p_count))
        pb_2d = path_blocked_by_planets(
            center_x_2d, center_y_2d, tgt_x_2d, tgt_y_2d, x, y, radius, active, margin=PATH_PLANET_MARGIN,
        )
        planet_blocks = jnp.broadcast_to(pb_2d[:, :, None], (p_count, p_count, bucket_axis))
    else:
        planet_blocks = jnp.zeros_like(sun_blocks, dtype=jnp.bool_)

    self_target = jnp.eye(planets.shape[0], dtype=jnp.bool_)
    target_valid_pair = target_valid[None, :]
    pair_valid = source_valid[:, None] & target_valid_pair

    bucket_valid = bucket_validity_mask(ship_counts, src_ships_grid)       # (P, P, B)
    full_valid = pair_valid[..., None] & bucket_valid & (~sun_blocks) & (~planet_blocks)

    from_ids = planets[:, 0]                         # (P,) float

    return {
        "source_valid": source_valid,
        "target_valid": target_valid,
        "angle": angle,
        "aim_x": aim_x,
        "aim_y": aim_y,
        "sun_blocks": sun_blocks,
        "planet_blocks": planet_blocks,
        "self_target": self_target,
        "target_valid_pair": target_valid_pair,
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
    """Return `(row (3,), mask_scalar)` for one move.

    Invalid moves emit a zero row and mask=0.
    """
    row = jnp.stack([
        from_id.astype(jnp.float32),
        angle.astype(jnp.float32),
        jnp.floor(ships).astype(jnp.float32),
    ])
    valid_f = valid.astype(jnp.float32)
    return row * valid_f, valid_f
