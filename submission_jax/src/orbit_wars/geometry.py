"""Pure JAX geometry helpers for Orbit Wars fleet/planet collision."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import BOARD_SIZE, CENTER, SUN_RADIUS


def distance_xy(x1: jnp.ndarray, y1: jnp.ndarray, x2: jnp.ndarray, y2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def point_to_segment_distance(
    px: jnp.ndarray,
    py: jnp.ndarray,
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
) -> jnp.ndarray:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    t = jnp.where(
        denom > 0.0,
        ((px - ax) * dx + (py - ay) * dy) / denom,
        0.0,
    )
    t = jnp.clip(t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    return jnp.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def swept_pair_hit(
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
    p0x: jnp.ndarray,
    p0y: jnp.ndarray,
    p1x: jnp.ndarray,
    p1y: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    disc = b * b - 4.0 * a * c
    no_motion = a < 1e-12
    hit_no_motion = c <= 0.0
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sq) / (2.0 * a + 1e-12)
    t2 = (-b + sq) / (2.0 * a + 1e-12)
    hit_motion = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(no_motion, hit_no_motion, hit_motion)


def fleet_speed(ships: jnp.ndarray, max_speed: jnp.ndarray) -> jnp.ndarray:
    log1000 = jnp.log(1000.0)
    speed = 1.0 + (max_speed - 1.0) * (jnp.log(jnp.maximum(ships, 1.0)) / log1000) ** 1.5
    return jnp.minimum(speed, max_speed)


def sun_hit(old_x, old_y, new_x, new_y) -> jnp.ndarray:
    return point_to_segment_distance(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS


def in_bounds(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE)
