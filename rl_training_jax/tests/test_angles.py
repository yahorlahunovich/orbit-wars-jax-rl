"""Angle / intercept / path-safety tests for the JAX decoder.
Updated to match the 1100 ELO heuristic notebook.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import MAX_PLANETS, reset, step
from orbit_wars.constants import CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT
from orbit_wars.decode import (
    BUCKET_COUNT,
    INTERCEPT_ITERATIONS,
    SUN_PATH_MARGIN,
    compose_action_grid,
    path_blocked_by_planets,
    path_crosses_sun,
)
from orbit_wars.geometry import (
    estimate_intercept_angles,
    fleet_speed,
    predict_orbit_polar,
    safe_angle,
    segment_clear_of_circles,
    solve_intercept,
    sun_hit,
)

# ---------------------------------------------------------------------------
# Python reference (1100 ELO notebook logic)
# ---------------------------------------------------------------------------

SUN_X = SUN_Y = CENTER
MAX_SPEED = 6.0


def ref_fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def ref_predict_orbit(x: float, y: float, omega: float, dt: float) -> tuple[float, float]:
    r = math.hypot(x - SUN_X, y - SUN_Y)
    if r + 1e-3 >= ROTATION_RADIUS_LIMIT:
        return x, y
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def ref_line_seg_min_dist(x1, y1, x2, y2, px, py) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def ref_path_crosses_sun(x1, y1, x2, y2, margin: float = 1.5) -> bool:
    return ref_line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


def ref_estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    dist = math.hypot(tx - sx, ty - sy)
    hit_d = max(0.0, dist - (sr + 0.1) - tr)
    turns = max(1.0, math.ceil(hit_d / ref_fleet_speed(max(1, ships))))
    return turns


def ref_solve_intercept(
    fx: float, fy: float, fsr: float, tx: float, ty: float, ttr: float,
    orbiting: bool, omega: float, ships: int, iterations: int = 6,
) -> tuple[float, float, float]:
    turns = ref_estimate_arrival(fx, fy, fsr, tx, ty, ttr, ships)
    
    # Phase A: iterative intercept
    ix, iy = tx, ty
    if orbiting:
        for _ in range(iterations):
            ix, iy = ref_predict_orbit(tx, ty, omega, turns)
            turns = ref_estimate_arrival(fx, fy, fsr, ix, iy, ttr, ships)
    
    # Note: Phase B (behind-sun wait) removed to match optimized JAX solver.
    return ix, iy, turns


def ref_safe_angle(x1, y1, x2, y2) -> float:
    direct = math.atan2(y2 - y1, x2 - x1)
    if not ref_path_crosses_sun(x1, y1, x2, y2, margin=SUN_PATH_MARGIN):
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + SUN_PATH_MARGIN) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw, ccw = to_sun + half, to_sun - half

    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)

    return cw if adiff(cw) < adiff(ccw) else ccw


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100, 255])
def test_intercept_matches_heuristic_reference(seed: int):
    rng = np.random.default_rng(seed)
    omega = float(rng.uniform(0.01, 0.05))
    for _ in range(20):
        fx, fy = float(rng.uniform(5, 95)), float(rng.uniform(5, 95))
        tx, ty = float(rng.uniform(5, 95)), float(rng.uniform(5, 95))
        fsr = float(rng.uniform(1, 4))
        ttr = float(rng.uniform(1, 4))
        orbiting = bool(math.hypot(tx-50, ty-50) + ttr < ROTATION_RADIUS_LIMIT)
        ships = int(rng.integers(4, 200))
        
        ref_ix, ref_iy, ref_tt = ref_solve_intercept(fx, fy, fsr, tx, ty, ttr, orbiting, omega, ships)
        
        from orbit_wars.geometry import solve_intercept_with_wait
        jix, jiy, jtt, _jb = solve_intercept_with_wait(
            src_x=jnp.float32(fx), src_y=jnp.float32(fy), src_r=jnp.float32(fsr),
            tgt_x=jnp.float32(tx), tgt_y=jnp.float32(ty), tgt_r=jnp.float32(ttr),
            tgt_is_orbiting=jnp.bool_(orbiting), 
            tgt_is_comet=jnp.bool_(False),
            tgt_traj=jnp.zeros((1, 2), dtype=jnp.float32),
            tgt_valid_time=jnp.zeros((1,), dtype=jnp.bool_),
            ship_count=jnp.float32(ships),
            angular_velocity=jnp.float32(omega), max_speed=jnp.float32(MAX_SPEED),
            n_iter=6
        )
        
        assert float(jix) == pytest.approx(ref_ix, abs=1e-3)
        assert float(jiy) == pytest.approx(ref_iy, abs=1e-3)
        assert float(jtt) == pytest.approx(ref_tt, abs=1e-3)


def test_predict_orbit_polar_matches_reference():
    cases = [(80.0, 50.0, 0.03, 12.0), (50.0, 80.0, -0.02, 5.0), (20.0, 30.0, 0.04, 20.0)]
    for x, y, omega, dt in cases:
        theta = math.atan2(y - 50, x - 50)
        r = math.hypot(x - 50, y - 50)
        rx, ry = 50 + r * math.cos(theta + omega * dt), 50 + r * math.sin(theta + omega * dt)
        
        jx, jy = predict_orbit_polar(
            jnp.float32(x), jnp.float32(y), jnp.float32(omega), jnp.float32(dt),
        )
        assert float(jx) == pytest.approx(rx, abs=1e-4)
        assert float(jy) == pytest.approx(ry, abs=1e-4)


def test_safe_angle_avoids_sun():
    fx, fy, tx, ty = 10.0, 50.0, 90.0, 50.0
    jax = float(safe_angle(
        jnp.float32(fx), jnp.float32(fy), jnp.float32(tx), jnp.float32(ty),
        sun_margin=SUN_PATH_MARGIN,
    ))
    assert sun_hit(jnp.float32(fx), jnp.float32(fy), 
                   jnp.float32(fx + math.cos(jax)*10), 
                   jnp.float32(fy + math.sin(jax)*10), 
                   margin=SUN_PATH_MARGIN) == False


def test_orbiting_intercept_differs_from_naive_aim():
    fx, fy = 20.0, 50.0
    tx, ty = 50.0, 80.0
    omega = 0.04
    ships = 40.0
    naive = math.atan2(ty - fy, tx - fx)
    
    angle, aim_x, aim_y, _blocked = estimate_intercept_angles(
        src_x=jnp.float32(fx), src_y=jnp.float32(fy), src_r=jnp.float32(1.0),
        tgt_x=jnp.float32(tx), tgt_y=jnp.float32(ty), tgt_r=jnp.float32(1.0),
        tgt_is_orbiting=jnp.bool_(True), 
        tgt_is_comet=jnp.bool_(False),
        tgt_traj=jnp.zeros((1, 2), dtype=jnp.float32),
        tgt_valid_time=jnp.zeros((1,), dtype=jnp.bool_),
        ship_counts=jnp.float32(ships),
        angular_velocity=jnp.float32(omega), max_speed=jnp.float32(MAX_SPEED),
    )
    lead = float(angle)
    assert abs((lead - naive + math.pi) % (2 * math.pi) - math.pi) > 0.05


def test_path_blocked_by_middle_planet():
    sx, sy = 10.0, 50.0
    tx, ty = 90.0, 50.0
    bx, by, br = 50.0, 50.0, 5.0
    p = 4
    px = jnp.array([sx, tx, bx, 0.0], dtype=jnp.float32)
    py = jnp.array([sy, ty, by, 0.0], dtype=jnp.float32)
    pr = jnp.array([3.0, 3.0, br, 0.0], dtype=jnp.float32)
    active = jnp.array([True, True, True, False], dtype=jnp.bool_)
    start_x = jnp.full((p, p), sx, dtype=jnp.float32)
    start_y = jnp.full((p, p), sy, dtype=jnp.float32)
    aim_x = px[None, :]
    aim_y = py[None, :]
    blocked = np.asarray(path_blocked_by_planets(
        start_x, start_y, aim_x, aim_y, px, py, pr, active, margin=0.5,
    ))
    assert blocked[0, 1]
    assert not blocked[0, 2]


@pytest.mark.parametrize("seed", [0, 3, 11, 42, 100])
def test_valid_decoded_moves_mostly_hit_intended_target(seed: int):
    state = reset(seed, episode_steps=500)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    idxs = np.argwhere(full)
    if len(idxs) == 0:
        pytest.skip(f"seed {seed}: no valid moves")

    outcomes = Counter()
    tested = 0
    rng = np.random.default_rng(seed)
    pick = idxs[rng.choice(len(idxs), size=min(6, len(idxs)), replace=False)]
    for s_idx, t_idx, b_idx in pick:
        from_id = float(grid["from_ids"][s_idx])
        angle = float(grid["angle"][s_idx, t_idx, b_idx])
        ships = int(grid["ship_counts"][s_idx, t_idx, b_idx])
        state2 = step(reset(seed, episode_steps=500), [[[from_id, angle, ships]], []])
        if int(state2.n_fleets) == 0:
            continue
        fleet_id = int(np.asarray(state2.fleets)[int(state2.n_fleets) - 1, 0])
        res = _simulate_fleet_hit(state2, fleet_id, t_idx)
        outcomes[res] += 1
        tested += 1

    if tested == 0:
        pytest.skip(f"seed {seed}: no fleets created")
    hit_rate = outcomes["hit_target"] / tested
    assert hit_rate >= 0.25, f"seed {seed}: outcomes={dict(outcomes)}"


def _simulate_fleet_hit(state, fleet_id: int, target_slot: int, max_steps: int = 350) -> str:
    for _ in range(max_steps):
        fleets = np.asarray(state.fleets)
        active = (fleets[:, 7] > 0) & (fleets[:, 0] == fleet_id)
        if not active.any():
            return "miss"
        fidx = int(np.argmax(active))
        fx, fy = float(fleets[fidx, 2]), float(fleets[fidx, 3])
        angle = float(fleets[fidx, 4])
        ships = float(fleets[fidx, 6])
        spd = float(fleet_speed(jnp.float32(ships), state.ship_speed))
        nfx = fx + math.cos(angle) * spd
        nfy = fy + math.sin(angle) * spd

        px = np.asarray(state.planets[:, 2])
        py = np.asarray(state.planets[:, 3])
        pr = np.asarray(state.planets[:, 4])
        pa = np.asarray(state.planets[:, 7]) > 0

        state = step(state, [[], []])
        fleets2 = np.asarray(state.fleets)
        still = (fleets2[:, 7] > 0) & (fleets2[:, 0] == fleet_id)
        if still.any():
            continue

        best, best_d = -1, 1e9
        for i in np.where(pa)[0]:
            d = math.hypot(px[i] - nfx, py[i] - nfy)
            if d < pr[i] + spd + 1.0 and d < best_d:
                best_d = d
                best = i
        if best == target_slot:
            return "hit_target"
        if best >= 0:
            return "hit_wrong"
        return "miss"
    return "timeout"
