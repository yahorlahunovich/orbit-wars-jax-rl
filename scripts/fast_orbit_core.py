from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_BUF = COMET_RADIUS + 0.5


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _point_segment_distance_py(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _swept_pair_hit_py(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
    radius: float,
) -> bool:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


@njit(cache=True)
def _point_segment_distance_nb(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    return math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))


@njit(cache=True)
def _swept_pair_hit_nb(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
    radius: float,
) -> bool:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


@njit(cache=True)
def move_fleets_core_numba(
    fleets: np.ndarray,
    planet_paths: np.ndarray,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move fleets and classify first collision.

    fleets columns:
        id, owner, x, y, angle, from_planet_id, ships
    planet_paths columns:
        planet_id, old_x, old_y, new_x, new_y, radius, check_collision

    Returns:
        new_xy[F, 2], remove_mask[F], hit_planet_index[F] (-1 for no planet hit)
    """
    n_fleets = fleets.shape[0]
    n_planets = planet_paths.shape[0]
    new_xy = np.empty((n_fleets, 2), dtype=np.float64)
    remove_mask = np.zeros(n_fleets, dtype=np.bool_)
    hit_planet_index = np.full(n_fleets, -1, dtype=np.int64)

    log1000 = math.log(1000.0)
    for i in range(n_fleets):
        angle = fleets[i, 4]
        ships = fleets[i, 6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / log1000) ** 1.5
        if speed > max_speed:
            speed = max_speed

        old_x = fleets[i, 2]
        old_y = fleets[i, 3]
        new_x = old_x + math.cos(angle) * speed
        new_y = old_y + math.sin(angle) * speed
        new_xy[i, 0] = new_x
        new_xy[i, 1] = new_y

        for j in range(n_planets):
            if planet_paths[j, 6] <= 0.0:
                continue
            if _swept_pair_hit_nb(
                old_x,
                old_y,
                new_x,
                new_y,
                planet_paths[j, 1],
                planet_paths[j, 2],
                planet_paths[j, 3],
                planet_paths[j, 4],
                planet_paths[j, 5],
            ):
                remove_mask[i] = True
                hit_planet_index[i] = j
                break
        if remove_mask[i]:
            continue

        if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
            remove_mask[i] = True
            continue

        if _point_segment_distance_nb(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS:
            remove_mask[i] = True

    return new_xy, remove_mask, hit_planet_index


def move_fleets_core_python(
    fleets: np.ndarray,
    planet_paths: np.ndarray,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fleets = fleets.shape[0]
    n_planets = planet_paths.shape[0]
    new_xy = np.empty((n_fleets, 2), dtype=np.float64)
    remove_mask = np.zeros(n_fleets, dtype=np.bool_)
    hit_planet_index = np.full(n_fleets, -1, dtype=np.int64)
    log1000 = math.log(1000.0)

    for i in range(n_fleets):
        angle = float(fleets[i, 4])
        ships = float(fleets[i, 6])
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / log1000) ** 1.5
        speed = min(speed, max_speed)
        old_x = float(fleets[i, 2])
        old_y = float(fleets[i, 3])
        new_x = old_x + math.cos(angle) * speed
        new_y = old_y + math.sin(angle) * speed
        new_xy[i, 0] = new_x
        new_xy[i, 1] = new_y

        for j in range(n_planets):
            if planet_paths[j, 6] <= 0.0:
                continue
            if _swept_pair_hit_py(
                old_x,
                old_y,
                new_x,
                new_y,
                float(planet_paths[j, 1]),
                float(planet_paths[j, 2]),
                float(planet_paths[j, 3]),
                float(planet_paths[j, 4]),
                float(planet_paths[j, 5]),
            ):
                remove_mask[i] = True
                hit_planet_index[i] = j
                break
        if remove_mask[i]:
            continue

        if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
            remove_mask[i] = True
            continue

        if _point_segment_distance_py(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS:
            remove_mask[i] = True

    return new_xy, remove_mask, hit_planet_index


def observation_to_arrays(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    initial_planets = list(_get(obs, "initial_planets", []) or [])
    comets = list(_get(obs, "comets", []) or [])
    comet_ids = {int(x) for x in (_get(obs, "comet_planet_ids", []) or [])}
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))
    step = int(_get(obs, "step", _get(obs, "turn", 0)))

    fleet_arr = np.asarray(fleets, dtype=np.float64).reshape((-1, 7))
    initial_by_id = {int(p[0]): p for p in initial_planets}
    rows: list[list[float]] = []

    for planet in planets:
        pid = int(planet[0])
        if pid in comet_ids:
            continue
        old_x = float(planet[2])
        old_y = float(planet[3])
        new_x = old_x
        new_y = old_y
        initial = initial_by_id.get(pid)
        if initial is not None:
            dx = float(initial[2]) - CENTER
            dy = float(initial[3]) - CENTER
            orbit_r = math.hypot(dx, dy)
            radius = float(planet[4])
            if orbit_r + radius < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy, dx)
                current_angle = initial_angle + angular_velocity * step
                new_x = CENTER + orbit_r * math.cos(current_angle)
                new_y = CENTER + orbit_r * math.sin(current_angle)
        rows.append([pid, old_x, old_y, new_x, new_y, float(planet[4]), 1.0])

    planet_by_id = {int(p[0]): p for p in planets}
    for group in comets:
        idx = int(group.get("path_index", -1)) + 1
        planet_ids = list(group.get("planet_ids") or [])
        paths = list(group.get("paths") or [])
        for i, pid_raw in enumerate(planet_ids):
            pid = int(pid_raw)
            planet = planet_by_id.get(pid)
            if planet is None or i >= len(paths):
                continue
            old_x = float(planet[2])
            old_y = float(planet[3])
            path = paths[i]
            if idx >= len(path):
                new_x, new_y = old_x, old_y
                check = 1.0
            else:
                new_x = float(path[idx][0])
                new_y = float(path[idx][1])
                check = 1.0 if old_x >= 0.0 else 0.0
            rows.append([pid, old_x, old_y, new_x, new_y, float(planet[4]), check])

    planet_arr = np.asarray(rows, dtype=np.float64).reshape((-1, 7))
    return fleet_arr, planet_arr


def warm_numba() -> None:
    fleets = np.array([[0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 20.0]], dtype=np.float64)
    planets = np.array([[0.0, 20.0, 10.0, 20.0, 10.0, 1.0, 1.0]], dtype=np.float64)
    move_fleets_core_numba(fleets, planets, 6.0)
    initial = np.array([[0.0, -1.0, 80.0, 80.0, 1.0, 10.0, 1.0]], dtype=np.float64)
    comet_ids = np.empty(0, dtype=np.int64)
    _generate_comet_visible_nb(0.8, 80.0, 0.7, initial, comet_ids, 0.03, 50, 4.0)


@njit(cache=True)
def _contains_int(values: np.ndarray, needle: int) -> bool:
    for i in range(values.shape[0]):
        if int(values[i]) == needle:
            return True
    return False


@njit(cache=True)
def _generate_comet_visible_nb(
    eccentricity: float,
    semi_major: float,
    phi: float,
    initial_planets: np.ndarray,
    comet_planet_ids: np.ndarray,
    angular_velocity: float,
    spawn_step: int,
    comet_speed: float,
) -> tuple[bool, np.ndarray, np.ndarray, int]:
    visible_x = np.empty(64, dtype=np.float64)
    visible_y = np.empty(64, dtype=np.float64)
    if semi_major * (1.0 - eccentricity) < SUN_RADIUS + COMET_RADIUS:
        return False, visible_x, visible_y, 0

    semi_minor = semi_major * math.sqrt(1.0 - eccentricity * eccentricity)
    c_val = semi_major * eccentricity
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    num = 5000
    path_x = np.empty(512, dtype=np.float64)
    path_y = np.empty(512, dtype=np.float64)
    path_n = 0

    t0 = 0.3 * math.pi
    ex0 = c_val + semi_major * math.cos(t0)
    ey0 = semi_minor * math.sin(t0)
    prev_x = CENTER + ex0 * cos_phi - ey0 * sin_phi
    prev_y = CENTER + ex0 * sin_phi + ey0 * cos_phi
    path_x[path_n] = prev_x
    path_y[path_n] = prev_y
    path_n += 1

    cum = 0.0
    target = comet_speed
    for i in range(1, num):
        t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
        ex = c_val + semi_major * math.cos(t)
        ey = semi_minor * math.sin(t)
        x = CENTER + ex * cos_phi - ey * sin_phi
        y = CENTER + ex * sin_phi + ey * cos_phi
        dx = x - prev_x
        dy = y - prev_y
        cum += math.sqrt(dx * dx + dy * dy)
        if cum >= target:
            if path_n >= path_x.shape[0]:
                return False, visible_x, visible_y, 0
            path_x[path_n] = x
            path_y[path_n] = y
            path_n += 1
            target += comet_speed
        prev_x = x
        prev_y = y

    board_start = -1
    board_end = -1
    for i in range(path_n):
        x = path_x[i]
        y = path_y[i]
        if 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE:
            if board_start < 0:
                board_start = i
            board_end = i
    if board_start < 0:
        return False, visible_x, visible_y, 0

    visible_n = board_end - board_start + 1
    if not (5 <= visible_n <= 40):
        return False, visible_x, visible_y, 0

    for i in range(visible_n):
        visible_x[i] = path_x[board_start + i]
        visible_y[i] = path_y[board_start + i]

    for k in range(visible_n):
        cx = visible_x[k]
        cy = visible_y[k]
        sun_dx = cx - CENTER
        sun_dy = cy - CENTER
        if math.sqrt(sun_dx * sun_dx + sun_dy * sun_dy) < SUN_RADIUS + COMET_RADIUS:
            return False, visible_x, visible_y, 0

        spx0 = cy
        spy0 = cx
        spx1 = BOARD_SIZE - cx
        spy1 = cy
        spx2 = cx
        spy2 = BOARD_SIZE - cy
        spx3 = BOARD_SIZE - cy
        spy3 = BOARD_SIZE - cx
        game_step = spawn_step - 1 + k

        for pidx in range(initial_planets.shape[0]):
            pid = int(initial_planets[pidx, 0])
            if _contains_int(comet_planet_ids, pid):
                continue
            px0 = initial_planets[pidx, 2]
            py0 = initial_planets[pidx, 3]
            pradius = initial_planets[pidx, 4]
            dx0 = px0 - CENTER
            dy0 = py0 - CENTER
            orbit_r = math.sqrt(dx0 * dx0 + dy0 * dy0)
            is_orbiting = orbit_r + pradius < ROTATION_RADIUS_LIMIT
            if is_orbiting:
                init_angle = math.atan2(dy0, dx0)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orbit_r * math.cos(cur_angle)
                py = CENTER + orbit_r * math.sin(cur_angle)
                limit = pradius + COMET_RADIUS
            else:
                px = px0
                py = py0
                limit = pradius + COMET_BUF

            dx = spx0 - px
            dy = spy0 - py
            if math.sqrt(dx * dx + dy * dy) < limit:
                return False, visible_x, visible_y, 0
            dx = spx1 - px
            dy = spy1 - py
            if math.sqrt(dx * dx + dy * dy) < limit:
                return False, visible_x, visible_y, 0
            dx = spx2 - px
            dy = spy2 - py
            if math.sqrt(dx * dx + dy * dy) < limit:
                return False, visible_x, visible_y, 0
            dx = spx3 - px
            dy = spy3 - py
            if math.sqrt(dx * dx + dy * dy) < limit:
                return False, visible_x, visible_y, 0

    return True, visible_x, visible_y, visible_n


def generate_comet_paths_fast(
    initial_planets: list[Any],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: set[int] | list[int] | None = None,
    comet_speed: float = 4.0,
    rng: Any = None,
) -> list[list[list[float]]] | None:
    if rng is None:
        import random

        rng = random
    comet_ids = (
        np.asarray(list(comet_planet_ids), dtype=np.int64)
        if comet_planet_ids is not None
        else np.empty(0, dtype=np.int64)
    )
    initial_arr = np.asarray(initial_planets, dtype=np.float64).reshape((-1, 7))

    for _ in range(300):
        eccentricity = rng.uniform(0.75, 0.93)
        semi_major = rng.uniform(60, 150)
        if semi_major * (1.0 - eccentricity) < SUN_RADIUS + COMET_RADIUS:
            continue
        phi = rng.uniform(math.pi / 6, math.pi / 3)
        ok, visible_x, visible_y, visible_n = _generate_comet_visible_nb(
            eccentricity,
            semi_major,
            phi,
            initial_arr,
            comet_ids,
            float(angular_velocity),
            int(spawn_step),
            float(comet_speed),
        )
        if not ok:
            continue
        visible = [(float(visible_x[i]), float(visible_y[i])) for i in range(visible_n)]
        return [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]
    return None
