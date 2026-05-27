"""Pure-JAX geometry decoder for Orbit Wars actions.

Given a chosen `(source_planet, target_planet, bucket)` triple, produce a
legal `[from_id, angle, num_ships]` action row plus validity flags. All
operations are vectorized and vmap-friendly — Phase 4 (rollout) will broadcast
over batch × source.

Ship-bucket scheme (BUCKET_COUNT = 8):

    0  10%  of source ships    (light probe)
    1  25%  of source ships
    2  50%  of source ships
    3  75%  of source ships
    4 100%  of source ships    (all-in)
    5  target_ships + 1        (minimal capture)
    6  target_ships + 50% src  (capture with reserve)
    7  4 ships                 (constant minimum)

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
from .geometry import point_to_segment_distance
from .state import OrbitWarsState

BUCKET_COUNT = 8

# Per-bucket coefficients: ship_count = max(MIN, src*src_frac + tgt*tgt_frac + plus)
_SRC_FRAC = jnp.array([0.10, 0.25, 0.50, 0.75, 1.00, 0.00, 0.50, 0.00], dtype=jnp.float32)
_TGT_FRAC = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.00], dtype=jnp.float32)
_PLUS = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 4.00], dtype=jnp.float32)
_MIN = jnp.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], dtype=jnp.float32)

# Launch offset to avoid spawning fleets inside the source planet.
LAUNCH_OFFSET_PADDING = 0.1


def ship_counts_for_buckets(
    source_ships: jnp.ndarray, target_ships: jnp.ndarray
) -> jnp.ndarray:
    """Return integer-valued ship counts for every bucket index.

    Inputs broadcast against each other. Output shape = broadcasted shape +
    `(BUCKET_COUNT,)`.
    """
    src = source_ships[..., None]
    tgt = target_ships[..., None]
    raw = src * _SRC_FRAC + tgt * _TGT_FRAC + _PLUS
    raw = jnp.maximum(raw, _MIN)
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
) -> jnp.ndarray:
    """Conservative check: does the straight segment src→tgt pass within
    SUN_RADIUS of the sun centre? Matches the env's `sun_hit` semantics.
    """
    d = point_to_segment_distance(
        jnp.float32(CENTER), jnp.float32(CENTER),
        src_x, src_y, tgt_x, tgt_y,
    )
    return d < SUN_RADIUS


def launch_angle(
    src_x: jnp.ndarray, src_y: jnp.ndarray,
    tgt_x: jnp.ndarray, tgt_y: jnp.ndarray,
) -> jnp.ndarray:
    """`atan2(dy, dx)` aim. Intercept correction lives in a higher layer."""
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)


def compose_action_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
) -> dict[str, jnp.ndarray]:
    """Pre-compute everything the policy/rollout needs about every
    (source, target, bucket) triple in a single state.

    Returns a dict with all (P_src, P_tgt) / (P_src, P_tgt, BUCKETS) shaped
    arrays:

        source_valid   (P,)             bool          source planet owned by player
        target_valid   (P,)             bool          target planet is active
        angle          (P, P)           float32       atan2 aim
        sun_blocks     (P, P)           bool          true if path passes through sun
        self_target    (P, P)           bool          true on the diagonal
        target_valid_pair (P, P)        bool          target_valid AND not self
        ship_counts    (P, P, BUCKETS)  float32       per-bucket ship count to send
        bucket_valid   (P, P, BUCKETS)  bool          ship count fits source's reserve
        pair_valid     (P, P)           bool          source_valid AND target_valid_pair AND NOT sun_blocks
        full_valid     (P, P, BUCKETS)  bool          pair_valid AND bucket_valid
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

    sx = x[:, None]                    # (P, 1)
    sy = y[:, None]
    tx = x[None, :]                    # (1, P)
    ty = y[None, :]

    angle = launch_angle(sx, sy, tx, ty)             # (P, P)
    # Compute the actual launch start position (just outside the source planet)
    # so the sun check matches what the env will see.
    start_x = sx + jnp.cos(angle) * (radius[:, None] + LAUNCH_OFFSET_PADDING)
    start_y = sy + jnp.sin(angle) * (radius[:, None] + LAUNCH_OFFSET_PADDING)
    sun_blocks = path_crosses_sun(start_x, start_y, tx, ty)

    self_target = jnp.eye(planets.shape[0], dtype=jnp.bool_)
    target_valid_pair = target_valid[None, :] & (~self_target)
    pair_valid = source_valid[:, None] & target_valid_pair & (~sun_blocks)

    src_ships_grid = ships[:, None]                  # (P, 1)
    tgt_ships_grid = ships[None, :]                  # (1, P)
    ship_counts = ship_counts_for_buckets(src_ships_grid, tgt_ships_grid)  # (P, P, B)
    bucket_valid = bucket_validity_mask(ship_counts, src_ships_grid)       # (P, P, B)
    full_valid = pair_valid[..., None] & bucket_valid

    from_ids = planets[:, 0]                         # (P,) float

    return {
        "source_valid": source_valid,
        "target_valid": target_valid,
        "angle": angle,
        "sun_blocks": sun_blocks,
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
