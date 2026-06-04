"""Pure JAX geometry helpers for Orbit Wars fleet/planet collision.
Updated with logic from the 1100 ELO heuristic notebook.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import BOARD_SIZE, CENTER, SUN_RADIUS


def _match_rank(arr: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Expand arr with trailing 1-dims to match ref's rank."""
    while arr.ndim < ref.ndim:
        arr = arr[..., None]
    return arr


def distance_xy(x1: jnp.ndarray, y1: jnp.ndarray, x2: jnp.ndarray, y2: jnp.ndarray) -> jnp.ndarray:
    dx = x1 - x2
    dy = y1 - y2
    return jnp.sqrt(dx * dx + dy * dy)


def in_bounds(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return (x >= 0) & (x <= BOARD_SIZE) & (y >= 0) & (y <= BOARD_SIZE)


def point_to_segment_distance(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
) -> jnp.ndarray:
    """Distance from point(s) (px, py) to line segment(s) (x1, y1) -> (x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy

    # Projection of point onto line (normalized to [0, 1])
    t = ((px - x1) * dx + (py - y1) * dy) / jnp.maximum(d2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return distance_xy(px, py, closest_x, closest_y)


def sun_hit(
    x1: jnp.ndarray, y1: jnp.ndarray, x2: jnp.ndarray, y2: jnp.ndarray, margin: float = 1.0
) -> jnp.ndarray:
    """Does the path from (x1, y1) to (x2, y2) hit the sun?"""
    d = point_to_segment_distance(
        jnp.float32(CENTER), jnp.float32(CENTER), x1, y1, x2, y2
    )
    return d <= (SUN_RADIUS + margin)


def is_orbiting_planet(x: jnp.ndarray, y: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    dx = x - CENTER
    dy = y - CENTER
    d = jnp.sqrt(dx * dx + dy * dy)
    # Match the threshold used in the official env (ROTATION_RADIUS_LIMIT = 50)
    return d + r < 50.0


def predict_orbit_polar(
    x: jnp.ndarray,
    y: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    turns_ahead: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = x - CENTER
    dy = y - CENTER
    theta = jnp.arctan2(dy, dx)
    orbit_r = jnp.sqrt(dx * dx + dy * dy)
    
    # Broadcast to match turns_ahead rank
    theta = _match_rank(theta, turns_ahead)
    orbit_r = _match_rank(orbit_r, turns_ahead)
    omega = _match_rank(angular_velocity, turns_ahead)

    theta2 = theta + omega * turns_ahead
    new_x = CENTER + orbit_r * jnp.cos(theta2)
    new_y = CENTER + orbit_r * jnp.sin(theta2)
    return new_x, new_y


def predict_planet_position(
    x: jnp.ndarray,
    y: jnp.ndarray,
    is_orbiting: jnp.ndarray,
    turns_ahead: jnp.ndarray,
    angular_velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    px, py = predict_orbit_polar(x, y, angular_velocity, turns_ahead)
    # Broadcast is_orbiting to match
    orb = _match_rank(is_orbiting, turns_ahead)
    xx = _match_rank(x, turns_ahead)
    yy = _match_rank(y, turns_ahead)
    return jnp.where(orb, px, xx), jnp.where(orb, py, yy)


def precompute_comet_trajectories(
    comet_active: jnp.ndarray,
    comet_pids: jnp.ndarray,
    comet_path_index: jnp.ndarray,
    comet_paths: jnp.ndarray,
    comet_path_lengths: jnp.ndarray,
    planet_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    MAX_LEN = comet_paths.shape[2]
    G = comet_active.shape[0]
    
    flat_active = jnp.repeat(comet_active, 4)
    flat_pids = comet_pids.reshape(-1)
    flat_paths = comet_paths.reshape(-1, MAX_LEN, 2)
    flat_plens = comet_path_lengths.reshape(-1)
    flat_group_idx = jnp.repeat(jnp.arange(G), 4)

    match_all = (flat_pids[None, :] == planet_ids[:, None]) & flat_active[None, :] & (planet_ids[:, None] >= 0)
    best_slot = jnp.argmax(match_all.astype(jnp.int32), axis=-1)  # (P,)
    is_comet = jnp.any(match_all, axis=-1)  # (P,)
    
    g_idx = jnp.take(flat_group_idx, best_slot)
    p_idx = best_slot 
    
    c_path_idx = jnp.take(comet_path_index, g_idx)  # (P,)
    
    t_range = jnp.arange(MAX_LEN)[None, :]  # (1, L)
    future_idx = c_path_idx[:, None] + t_range  # (P, L)
    safe_future_idx = jnp.clip(future_idx, 0, MAX_LEN - 1)
    
    trajectories = flat_paths[p_idx[:, None], safe_future_idx]  # (P, L, 2)
    plen = flat_plens[p_idx]  # (P,)
    valid_time = (future_idx < plen[:, None]) & (future_idx >= 0)  # (P, L)
    
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
    safe_idx = jnp.clip(turns_ahead.astype(jnp.int32), 0, MAX_LEN - 1)
    
    orig_shape = safe_idx.shape
    flat_idx = safe_idx.reshape(-1)
    flat_traj = tgt_traj.reshape(-1, MAX_LEN, 2)
    
    res = flat_traj[jnp.arange(flat_idx.shape[0]), flat_idx]
    cx = res[:, 0].reshape(orig_shape)
    cy = res[:, 1].reshape(orig_shape)
    
    # 2. Orbital path
    ox, oy = predict_planet_position(tgt_x, tgt_y, tgt_is_orbiting, turns_ahead, angular_velocity)
    
    is_com = _match_rank(tgt_is_comet, turns_ahead)
    return jnp.where(is_com, cx, ox), jnp.where(is_com, cy, oy)


def get_arrival_turns(
    sx: jnp.ndarray, sy: jnp.ndarray, sr: jnp.ndarray,
    tx: jnp.ndarray, ty: jnp.ndarray, tr: jnp.ndarray,
    ships: jnp.ndarray, max_speed: jnp.ndarray,
) -> jnp.ndarray:
    d = distance_xy(sx, sy, tx, ty)
    tr = _match_rank(tr, d)
    sr = _match_rank(sr, d)
    
    hit_d = jnp.maximum(0.0, d - (sr + 0.1) - tr)
    speed = fleet_speed(ships, max_speed)
    
    hit_d_b = _match_rank(hit_d, speed)
    return jnp.maximum(1.0, jnp.ceil(hit_d_b / jnp.maximum(speed, 1e-6)))


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
    
    sx_b = _match_rank(sx, aim_x)
    sy_b = _match_rank(sy, aim_y)
    blocked = sun_hit(sx_b, sy_b, aim_x, aim_y, margin=sun_margin)

    return aim_x, aim_y, turns, blocked


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
    f_min_x = jnp.minimum(ax, bx)
    f_max_x = jnp.maximum(ax, bx)
    f_min_y = jnp.minimum(ay, by)
    f_max_y = jnp.maximum(ay, by)
    
    p_min_x = jnp.minimum(p0x, p1x) - radius
    p_max_x = jnp.maximum(p0x, p1x) + radius
    p_min_y = jnp.minimum(p0y, p1y) - radius
    p_max_y = jnp.maximum(p0y, p1y) + radius
    
    intersect = (f_min_x <= p_max_x) & (f_max_x >= p_min_x) & \
                (f_min_y <= p_max_y) & (f_max_y >= p_min_y)

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
    return intersect & jnp.where(no_motion, hit_no_motion, hit_motion)


def fleet_speed(ships: jnp.ndarray, max_speed: jnp.ndarray) -> jnp.ndarray:
    log_ships = jnp.log(jnp.maximum(ships, 1.0))
    log1000 = jnp.log(1000.0)
    speed = max_speed * (1.0 - 0.5 * jnp.minimum(1.0, log_ships / log1000))
    return jnp.maximum(speed, 1.0)
