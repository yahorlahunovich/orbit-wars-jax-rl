"""Pure JAX geometry helpers for Orbit Wars fleet/planet collision."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS


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


def sun_hit(old_x, old_y, new_x, new_y, margin: float = 0.0) -> jnp.ndarray:
    return point_to_segment_distance(
        jnp.float32(CENTER), jnp.float32(CENTER), old_x, old_y, new_x, new_y,
    ) < (SUN_RADIUS + margin)


def segment_intersects_circle(
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    d = point_to_segment_distance(cx, cy, ax, ay, bx, by)
    return d <= radius


def segment_clear_of_circles(
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    bx: jnp.ndarray,
    by: jnp.ndarray,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    radius: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """True when the segment does not intersect any valid circle."""
    blocked = segment_intersects_circle(ax, ay, bx, by, cx, cy, radius)
    return ~jnp.any(blocked & valid, axis=-1)


def _angle_diff(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    dd = (a - b) % (2.0 * jnp.pi)
    return jnp.minimum(dd, 2.0 * jnp.pi - dd)


def safe_angle(
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    aim_x: jnp.ndarray,
    aim_y: jnp.ndarray,
    sun_margin: float = 1.5,
) -> jnp.ndarray:
    """Return a launch angle from source to aim, detouring around the sun if needed."""
    src_x = src_x.astype(jnp.float64)
    src_y = src_y.astype(jnp.float64)
    aim_x = aim_x.astype(jnp.float64)
    aim_y = aim_y.astype(jnp.float64)
    margin = jnp.float64(sun_margin)
    sun_r = jnp.float64(SUN_RADIUS)
    center = jnp.float64(CENTER)

    direct = jnp.arctan2(aim_y - src_y, aim_x - src_x)
    crosses = sun_hit(src_x, src_y, aim_x, aim_y, margin=float(sun_margin))
    d = jnp.sqrt((src_x - center) ** 2 + (src_y - center) ** 2)
    inside = d <= sun_r + 1.0
    half = jnp.arcsin(jnp.minimum(1.0, (sun_r + margin) / jnp.maximum(d, 1e-6)))
    to_sun = jnp.arctan2(center - src_y, center - src_x)
    cw = to_sun + half
    ccw = to_sun - half
    detour = jnp.where(_angle_diff(cw, direct) <= _angle_diff(ccw, direct), cw, ccw)
    return jnp.where(crosses & ~inside, detour, direct).astype(jnp.float32)


def predict_orbit_polar(
    x: jnp.ndarray,
    y: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    turns_ahead: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Match heuristic notebook: advance polar angle around the sun."""
    theta = jnp.arctan2(y - CENTER, x - CENTER)
    r = jnp.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
    theta2 = theta + angular_velocity * turns_ahead
    return CENTER + r * jnp.cos(theta2), CENTER + r * jnp.sin(theta2)


def in_bounds(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE)


def rotate_around_center(x: jnp.ndarray, y: jnp.ndarray, radians: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = x - CENTER
    dy = y - CENTER
    c = jnp.cos(radians)
    s = jnp.sin(radians)
    return CENTER + dx * c - dy * s, CENTER + dx * s + dy * c


def is_orbiting_planet(x: jnp.ndarray, y: jnp.ndarray, radius: jnp.ndarray) -> jnp.ndarray:
    orbit_r = jnp.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
    return orbit_r + radius < ROTATION_RADIUS_LIMIT


def predict_planet_position(
    x: jnp.ndarray,
    y: jnp.ndarray,
    is_orbiting: jnp.ndarray,
    turns_ahead: jnp.ndarray,
    angular_velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    px, py = predict_orbit_polar(x, y, angular_velocity, turns_ahead)
    return jnp.where(is_orbiting, px, x), jnp.where(is_orbiting, py, y)


def solve_intercept(
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    tgt_x: jnp.ndarray,
    tgt_y: jnp.ndarray,
    tgt_is_orbiting: jnp.ndarray,
    ship_count: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    max_speed: jnp.ndarray,
    n_iter: int = 25,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return `(aim_x, aim_y, travel_time)` for a fleet of `ship_count` ships."""
    speed = fleet_speed(ship_count, max_speed)
    dist0 = distance_xy(src_x, src_y, tgt_x, tgt_y)
    travel_time = dist0 / jnp.maximum(speed, 1e-6)

    def body(_i, carry):
        tt, _ix, _iy = carry
        ix, iy = predict_planet_position(tgt_x, tgt_y, tgt_is_orbiting, tt, angular_velocity)
        tt_new = distance_xy(src_x, src_y, ix, iy) / jnp.maximum(speed, 1e-6)
        return tt_new, ix, iy

    ix0, iy0 = predict_planet_position(tgt_x, tgt_y, tgt_is_orbiting, travel_time, angular_velocity)
    travel_time, aim_x, aim_y = jax.lax.fori_loop(0, n_iter, body, (travel_time, ix0, iy0))
    return aim_x, aim_y, travel_time


def estimate_intercept_angles(
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    tgt_x: jnp.ndarray,
    tgt_y: jnp.ndarray,
    tgt_is_orbiting: jnp.ndarray,
    ship_counts: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    max_speed: jnp.ndarray,
    n_iter: int = 25,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Iterative lead-angle estimate matching the heuristic bot.

    All inputs broadcast to a common shape ending with optional bucket dim.
    Returns `(angle, aim_x, aim_y)` with the same broadcast shape.
    """
    speed = fleet_speed(ship_counts, max_speed)
    dist0 = distance_xy(src_x, src_y, tgt_x, tgt_y)
    travel_time = dist0 / jnp.maximum(speed, 1e-6)
    pred_x, pred_y = predict_planet_position(
        tgt_x, tgt_y, tgt_is_orbiting, travel_time, angular_velocity,
    )

    def body(_i, carry):
        px, py, _tt = carry
        dist = distance_xy(src_x, src_y, px, py)
        tt_new = dist / jnp.maximum(speed, 1e-6)
        nx, ny = predict_planet_position(tgt_x, tgt_y, tgt_is_orbiting, tt_new, angular_velocity)
        return nx, ny, tt_new

    pred_x, pred_y, _ = jax.lax.fori_loop(0, n_iter, body, (pred_x, pred_y, travel_time))
    angle = jnp.arctan2(pred_y - src_y, pred_x - src_x)
    return angle, pred_x, pred_y
