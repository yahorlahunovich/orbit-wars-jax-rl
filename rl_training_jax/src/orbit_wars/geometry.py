"""Pure JAX geometry helpers for Orbit Wars fleet/planet collision.
Updated with logic from the 1100 ELO heuristic notebook.
"""

from __future__ import annotations

from typing import Any
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
    log_ships = jnp.log(jnp.maximum(ships, 1.0))
    log1000 = jnp.log(1000.0)
    ratio = jnp.clip(log_ships / log1000, 0.0, 1.0)
    speed = 1.0 + (max_speed - 1.0) * (ratio ** 1.5)
    return speed


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
    src_x = jnp.asarray(src_x).astype(jnp.float32)
    src_y = jnp.asarray(src_y).astype(jnp.float32)
    aim_x = jnp.asarray(aim_x).astype(jnp.float32)
    aim_y = jnp.asarray(aim_y).astype(jnp.float32)
    margin = jnp.float32(sun_margin)
    sun_r = jnp.float32(SUN_RADIUS)
    center = jnp.float32(CENTER)

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
    theta = jnp.arctan2(y - CENTER, x - CENTER)
    r = jnp.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
    theta2 = theta + angular_velocity * turns_ahead
    nx, ny = CENTER + r * jnp.cos(theta2), CENTER + r * jnp.sin(theta2)
    return nx, ny


def in_bounds(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return (x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE)


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


def precompute_comet_trajectories(
    comet_active: jnp.ndarray,
    comet_pids: jnp.ndarray,
    comet_path_index: jnp.ndarray,
    comet_paths: jnp.ndarray,
    comet_path_lengths: jnp.ndarray,
    planet_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute comet paths for all planets to avoid expensive lookups inside geometry loops.
    
    Returns:
        is_comet: (P,) boolean mask
        trajectories: (P, MAX_COMET_PATH_LEN, 2)
    """
    MAX_LEN = comet_paths.shape[2]
    P = planet_ids.shape[0]
    G = comet_active.shape[0]
    
    flat_active = jnp.repeat(comet_active, 4)
    flat_pids = comet_pids.reshape(-1)
    flat_paths = comet_paths.reshape(-1, MAX_LEN, 2)
    flat_plens = comet_path_lengths.reshape(-1)
    flat_group_idx = jnp.repeat(jnp.arange(G), 4)

    # (P, 32)
    match_all = (flat_pids[None, :] == planet_ids[:, None]) & flat_active[None, :] & (planet_ids[:, None] >= 0)
    best_slot = jnp.argmax(match_all.astype(jnp.int32), axis=-1)  # (P,)
    is_comet = jnp.any(match_all, axis=-1)  # (P,)
    
    g_idx = jnp.take(flat_group_idx, best_slot)
    p_idx = best_slot 
    
    c_path_idx = jnp.take(comet_path_index, g_idx)  # (P,)
    
    # We want a tensor of shape (P, MAX_LEN, 2) that can be easily indexed by turns_ahead
    # We can just construct it explicitly.
    t_range = jnp.arange(MAX_LEN)[None, :]  # (1, L)
    future_idx = c_path_idx[:, None] + t_range  # (P, L)
    safe_future_idx = jnp.clip(future_idx, 0, MAX_LEN - 1)
    
    # Extract
    trajectories = flat_paths[p_idx[:, None], safe_future_idx]  # (P, L, 2)
    
    # Mask out invalid times
    plen = flat_plens[p_idx]  # (P,)
    valid_time = (future_idx < plen[:, None]) & (future_idx >= 0)  # (P, L)
    
    # Where invalid, we'll just put infinity to ensure it's not used,
    # though it shouldn't matter if is_comet masks it.
    # Actually, let's leave it as is, we'll mask by valid_time in the fast predictor.
    
    return is_comet, trajectories, valid_time


def predict_target_position_fast(
    tgt_x: jnp.ndarray,
    tgt_y: jnp.ndarray,
    tgt_is_orbiting: jnp.ndarray,
    tgt_is_comet: jnp.ndarray,
    tgt_traj: jnp.ndarray,
    tgt_valid_time: jnp.ndarray,
    turns_ahead: jnp.ndarray,
    angular_velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    MAX_LEN = tgt_traj.shape[-2]
    
    # 1. Comet path
    safe_idx = jnp.clip(turns_ahead.astype(jnp.int32), 0, MAX_LEN - 1)
    
    # tgt_traj is (P, P, B, MAX_LEN, 2)
    # We want to index into the second to last dimension.
    # We can do this efficiently by flattening the spatial dimensions.
    orig_shape = safe_idx.shape
    flat_idx = safe_idx.reshape(-1)
    flat_traj = tgt_traj.reshape(-1, MAX_LEN, 2)
    
    # Advanced indexing:
    flat_c = flat_traj[jnp.arange(flat_idx.shape[0]), flat_idx]
    
    cx = flat_c[:, 0].reshape(orig_shape)
    cy = flat_c[:, 1].reshape(orig_shape)
    
    # 2. Orbital path
    ox, oy = predict_planet_position(tgt_x, tgt_y, tgt_is_orbiting, turns_ahead, angular_velocity)
    
    return jnp.where(tgt_is_comet, cx, ox), jnp.where(tgt_is_comet, cy, oy)


def get_arrival_turns(
    sx: jnp.ndarray, sy: jnp.ndarray, sr: jnp.ndarray,
    tx: jnp.ndarray, ty: jnp.ndarray, tr: jnp.ndarray,
    ships: jnp.ndarray, max_speed: jnp.ndarray,
) -> jnp.ndarray:
    d = distance_xy(sx, sy, tx, ty)
    hit_d = jnp.maximum(0.0, d - (sr + 0.1) - tr)
    speed = fleet_speed(ships, max_speed)
    return jnp.maximum(1.0, jnp.ceil(hit_d / jnp.maximum(speed, 1e-6)))


def solve_intercept_with_wait(
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    src_r: jnp.ndarray,
    tgt_x: jnp.ndarray,
    tgt_y: jnp.ndarray,
    tgt_r: jnp.ndarray,
    tgt_is_orbiting: jnp.ndarray,
    tgt_is_comet: jnp.ndarray,
    tgt_traj: jnp.ndarray,
    tgt_valid_time: jnp.ndarray,
    ship_count: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    max_speed: jnp.ndarray,
    sun_margin: float = 1.5,
    n_iter: int = 6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    sx = jnp.asarray(src_x).astype(jnp.float32)
    sy = jnp.asarray(src_y).astype(jnp.float32)
    sr = jnp.asarray(src_r).astype(jnp.float32)
    tx = jnp.asarray(tgt_x).astype(jnp.float32)
    ty = jnp.asarray(tgt_y).astype(jnp.float32)
    tr = jnp.asarray(tgt_r).astype(jnp.float32)
    is_orb = jnp.asarray(tgt_is_orbiting).astype(jnp.bool_)
    is_com = jnp.asarray(tgt_is_comet).astype(jnp.bool_)
    count = jnp.asarray(ship_count).astype(jnp.float32)
    speed = jnp.asarray(max_speed).astype(jnp.float32)
    
    turns = get_arrival_turns(sx, sy, sr, tx, ty, tr, count, speed)

    def body(_i, carry):
        tt, _ix, _iy = carry
        ix, iy = predict_target_position_fast(tx, ty, is_orb, is_com, tgt_traj, tgt_valid_time, tt, angular_velocity)
        tt_new = get_arrival_turns(sx, sy, sr, ix, iy, tr, count, speed)
        return tt_new, ix, iy

    ix0, iy0 = predict_target_position_fast(tx, ty, is_orb, is_com, tgt_traj, tgt_valid_time, turns, angular_velocity)
    turns, aim_x, aim_y = jax.lax.fori_loop(0, n_iter, body, (turns, ix0, iy0))
    blocked = sun_hit(sx, sy, aim_x, aim_y, margin=sun_margin)

    def try_future_wait(carry, wait_t):
        best_aim_x, best_aim_y, best_turns, currently_blocked = carry
        fx, fy = predict_target_position_fast(tx, ty, is_orb, is_com, tgt_traj, tgt_valid_time, wait_t, angular_velocity)
        f_blocked = sun_hit(sx, sy, fx, fy, margin=sun_margin)
        f_turns = get_arrival_turns(sx, sy, sr, fx, fy, tr, count, speed)
        should_update = currently_blocked & (~f_blocked)
        return (
            jnp.where(should_update, fx, best_aim_x),
            jnp.where(should_update, fy, best_aim_y),
            jnp.where(should_update, f_turns, best_turns),
            currently_blocked & f_blocked
        ), None

    wait_times = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=jnp.float32)
    (aim_x, aim_y, turns, final_blocked), _ = jax.lax.scan(try_future_wait, (aim_x, aim_y, turns, blocked), wait_times)
    return aim_x, aim_y, turns, final_blocked


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
    aim_x, aim_y, turns, _blocked = solve_intercept_with_wait(
        src_x, src_y, 0.0, 
        tgt_x, tgt_y, 0.0, tgt_is_orbiting, jnp.zeros_like(tgt_is_orbiting),
        jnp.zeros((tgt_x.shape + (1, 2))), jnp.zeros((tgt_x.shape + (1,))),
        ship_count, angular_velocity, max_speed, n_iter=n_iter
    )
    return aim_x, aim_y, turns


def estimate_intercept_angles(
    src_x: jnp.ndarray,
    src_y: jnp.ndarray,
    src_r: jnp.ndarray,
    tgt_x: jnp.ndarray,
    tgt_y: jnp.ndarray,
    tgt_r: jnp.ndarray,
    tgt_is_orbiting: jnp.ndarray,
    tgt_is_comet: jnp.ndarray,
    tgt_traj: jnp.ndarray,
    tgt_valid_time: jnp.ndarray,
    ship_counts: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    max_speed: jnp.ndarray,
    n_iter: int = 6,
    sun_margin: float = 1.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    aim_x, aim_y, _turns, blocked = solve_intercept_with_wait(
        src_x, src_y, src_r, tgt_x, tgt_y, tgt_r, tgt_is_orbiting,
        tgt_is_comet, tgt_traj, tgt_valid_time,
        ship_counts, angular_velocity, max_speed, n_iter=n_iter,
        sun_margin=sun_margin,
    )
    angle = jnp.arctan2(aim_y - src_y, aim_x - src_x)
    return angle, aim_x, aim_y, blocked
