"""Comet spawn logic (vendored from official orbit_wars.py — no kaggle import needed)."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_comet_paths(
    initial_planets: list[list[float]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: list[int] | set[int] | None = None,
    comet_speed: float = 4.0,
    rng: random.Random | None = None,
) -> list[list[list[float]]] | None:
    """Generate 4 symmetric elliptical comet paths (matches official env)."""
    if rng is None:
        rng = random.Random()
    comet_ids = set(comet_planet_ids or [])

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense: list[tuple[float, float]] = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets: list[list[float]] = []
        orbiting_planets: list[list[float]] = []
        for planet in initial_planets:
            if planet[0] in comet_ids:
                continue
            pr = _distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def spawn_comet_for_state(
    planets: np.ndarray,
    n_planets: int,
    initial_planets: np.ndarray,
    comet_planet_ids: np.ndarray,
    n_comet_ids: int,
    angular_velocity: float,
    spawn_step: int,
    episode_seed: int,
    comet_speed: float = 4.0,
) -> dict[str, Any] | None:
    """Return dict with new planet rows + comet group, or None if spawn fails."""
    planet_rows = [
        [float(planets[i, j]) for j in range(7)]
        for i in range(int(n_planets))
        if planets[i, 7] > 0.0
    ]
    initial_rows = [
        [float(initial_planets[i, j]) for j in range(7)]
        for i in range(int(n_planets))
        if initial_planets[i, 7] > 0.0
    ]
    comet_ids = [int(comet_planet_ids[i]) for i in range(int(n_comet_ids)) if int(comet_planet_ids[i]) >= 0]
    comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
    paths = generate_comet_paths(
        initial_rows,
        float(angular_velocity),
        int(spawn_step),
        comet_ids,
        float(comet_speed),
        rng=comet_rng,
    )
    if not paths:
        return None

    next_id = max(int(p[0]) for p in planet_rows) + 1
    comet_ships = min(
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
    )
    new_planets: list[list[float]] = []
    group_pids: list[int] = []
    for i, _path in enumerate(paths):
        pid = next_id + i
        group_pids.append(pid)
        new_planets.append([pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION])
    return {
        "new_planets": new_planets,
        "group": {"planet_ids": group_pids, "paths": paths, "path_index": -1},
        "new_comet_ids": group_pids,
    }
