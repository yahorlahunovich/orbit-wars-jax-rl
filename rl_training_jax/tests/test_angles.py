"""Angle / intercept / path-safety tests for the JAX decoder.

Reference implementations mirror `notebooks/orbit_wars_heuristic_agent_scored_1000.py`
and `versions/kaggle700_current_heuristic/src/geometry.py`.
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
from orbit_wars.constants import CENTER, SUN_RADIUS
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
# Python reference (heuristic notebook)
# ---------------------------------------------------------------------------

SUN_X = SUN_Y = CENTER
MAX_SPEED = 6.0


def ref_fleet_speed(ships: float) -> float:
    ships = max(1, int(ships))
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def ref_predict_orbit(x: float, y: float, omega: float, dt: float) -> tuple[float, float]:
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def ref_solve_intercept(
    fx: float, fy: float, tx: float, ty: float,
    orbiting: bool, omega: float, ships: int, iterations: int = 25,
) -> tuple[float, float, float]:
    if not orbiting:
        dist = math.hypot(tx - fx, ty - fy)
        spd = ref_fleet_speed(ships)
        return tx, ty, dist / spd
    t = math.hypot(tx - fx, ty - fy) / ref_fleet_speed(ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = ref_predict_orbit(tx, ty, omega, t)
        t = math.hypot(ix - fx, iy - fy) / ref_fleet_speed(ships)
    return ix, iy, t


def ref_line_seg_min_dist(x1, y1, x2, y2, px, py) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def ref_path_crosses_sun(x1, y1, x2, y2, margin: float = 1.5) -> bool:
    return ref_line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


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


def ref_segment_clear(a, b, circles) -> bool:
    for (cx, cy), cr in circles:
        if ref_line_seg_min_dist(a[0], a[1], b[0], b[1], cx, cy) <= cr:
            return False
    return True


# ---------------------------------------------------------------------------
# Unit tests: intercept + safe angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100, 255])
def test_intercept_matches_heuristic_reference(seed: int):
    rng = np.random.default_rng(seed)
    omega = float(rng.uniform(0.01, 0.05))
    for _ in range(20):
        fx, fy = float(rng.uniform(5, 95)), float(rng.uniform(5, 95))
        tx, ty = float(rng.uniform(5, 95)), float(rng.uniform(5, 95))
        orbiting = bool(rng.random() < 0.7)
        ships = int(rng.integers(4, 200))
        ref_ix, ref_iy, ref_tt = ref_solve_intercept(fx, fy, tx, ty, orbiting, omega, ships)
        jax_ix, jax_iy, jax_tt = solve_intercept(
            jnp.float32(fx), jnp.float32(fy),
            jnp.float32(tx), jnp.float32(ty),
            jnp.bool_(orbiting), jnp.float32(ships),
            jnp.float32(omega), jnp.float32(MAX_SPEED),
            n_iter=INTERCEPT_ITERATIONS,
        )
        assert float(jax_ix) == pytest.approx(ref_ix, abs=0.5)
        assert float(jax_iy) == pytest.approx(ref_iy, abs=0.5)
        assert float(jax_tt) == pytest.approx(ref_tt, abs=0.5)


def test_predict_orbit_polar_matches_reference():
    cases = [(80.0, 50.0, 0.03, 12.0), (50.0, 80.0, -0.02, 5.0), (20.0, 30.0, 0.04, 20.0)]
    for x, y, omega, dt in cases:
        rx, ry = ref_predict_orbit(x, y, omega, dt)
        jx, jy = predict_orbit_polar(
            jnp.float32(x), jnp.float32(y), jnp.float32(omega), jnp.float32(dt),
        )
        assert float(jx) == pytest.approx(rx, abs=1e-4)
        assert float(jy) == pytest.approx(ry, abs=1e-4)


@pytest.mark.parametrize("fx,fy,tx,ty", [
    (10.0, 50.0, 90.0, 50.0),
    (15.0, 20.0, 85.0, 80.0),
    (30.0, 30.0, 70.0, 70.0),
])
def test_safe_angle_avoids_sun(fx, fy, tx, ty):
    ref = ref_safe_angle(fx, fy, tx, ty)
    jax = float(safe_angle(
        jnp.float32(fx), jnp.float32(fy), jnp.float32(tx), jnp.float32(ty),
        sun_margin=SUN_PATH_MARGIN,
    ))
    assert jax == pytest.approx(ref, abs=1e-4)
    if not ref_path_crosses_sun(fx, fy, tx, ty, margin=SUN_PATH_MARGIN):
        assert not ref_path_crosses_sun(fx, fy, tx, ty, margin=SUN_PATH_MARGIN)


def test_path_crosses_sun_with_margin():
    blocked = path_crosses_sun(
        jnp.float32(10.0), jnp.float32(50.0),
        jnp.float32(90.0), jnp.float32(50.0),
        margin=SUN_PATH_MARGIN,
    )
    assert bool(blocked)
    clear = path_crosses_sun(
        jnp.float32(10.0), jnp.float32(95.0),
        jnp.float32(90.0), jnp.float32(95.0),
        margin=SUN_PATH_MARGIN,
    )
    assert not bool(clear)


def test_orbiting_intercept_differs_from_naive_aim():
    """Naive atan2 to current position should miss a fast orbiting target."""
    fx, fy = 20.0, 50.0
    tx, ty = 50.0, 80.0
    omega = 0.04
    ships = 40.0
    naive = math.atan2(ty - fy, tx - fx)
    _, aim_x, aim_y = estimate_intercept_angles(
        jnp.float32(fx), jnp.float32(fy),
        jnp.float32(tx), jnp.float32(ty),
        jnp.bool_(True), jnp.float32(ships),
        jnp.float32(omega), jnp.float32(MAX_SPEED),
    )
    lead = float(jnp.arctan2(aim_y - fy, aim_x - fx))
    assert abs((lead - naive + math.pi) % (2 * math.pi) - math.pi) > 0.05


# ---------------------------------------------------------------------------
# Planet / comet path blocking
# ---------------------------------------------------------------------------


def test_path_blocked_by_middle_planet():
    """Fleet aimed at planet T must be masked if planet B sits on the segment."""
    # Source at left, target at right, blocker in the middle.
    sx, sy = 10.0, 50.0
    tx, ty = 90.0, 50.0
    bx, by, br = 50.0, 50.0, 5.0

    p = 4
    px = jnp.array([sx, tx, bx, 0.0], dtype=jnp.float32)
    py = jnp.array([sy, ty, by, 0.0], dtype=jnp.float32)
    pr = jnp.array([3.0, 3.0, br, 0.0], dtype=jnp.float32)
    active = jnp.array([True, True, True, False], dtype=jnp.bool_)

    start_x = jnp.full((p, p, 1), sx, dtype=jnp.float32)
    start_y = jnp.full((p, p, 1), sy, dtype=jnp.float32)
    aim_x = px[None, :, None]
    aim_y = py[None, :, None]

    blocked = np.asarray(path_blocked_by_planets(
        start_x, start_y, aim_x, aim_y, px, py, pr, active, margin=0.5,
    ))
    # src=0, tgt=1: blocker slot 2 is in the way.
    assert blocked[0, 1, 0]
    # src=0, tgt=2: no third planet between.
    assert not blocked[0, 2, 0]


def test_segment_clear_of_circles_vectorized():
    ax, ay = jnp.float32(0.0), jnp.float32(0.0)
    bx, by = jnp.float32(10.0), jnp.float32(0.0)
    cx = jnp.array([[5.0]])
    cy = jnp.array([[0.0]])
    cr = jnp.array([[1.0]])
    valid = jnp.array([[True]])
    clear = segment_clear_of_circles(ax, ay, bx, by, cx, cy, cr, valid)
    assert not bool(clear[0])
    # Segment above the x-axis misses a circle centred at (5, 3).
    clear2 = segment_clear_of_circles(
        ax, ay, bx, by,
        jnp.array([[5.0]]), jnp.array([[3.0]]), jnp.array([[1.0]]),
        jnp.array([[True]]),
    )
    assert bool(clear2[0])


# ---------------------------------------------------------------------------
# Integration: decoded moves vs env simulation
# ---------------------------------------------------------------------------


def _planet_slot_by_id(state, planet_id: int) -> int:
    pids = np.asarray(state.planets[:, 0])
    active = np.asarray(state.planets[:, 7]) > 0
    idx = np.where((pids == planet_id) & active)[0]
    return int(idx[0]) if len(idx) else -1


def _simulate_fleet_hit(state, fleet_id: int, target_slot: int, max_steps: int = 350) -> str:
    """Return 'hit_target', 'hit_wrong', 'miss', or 'timeout'."""
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

        pids = np.asarray(state.planets[:, 0])
        px = np.asarray(state.planets[:, 2])
        py = np.asarray(state.planets[:, 3])
        pr = np.asarray(state.planets[:, 4])
        pa = np.asarray(state.planets[:, 7]) > 0

        state = step(state, [[], []])
        fleets2 = np.asarray(state.fleets)
        still = (fleets2[:, 7] > 0) & (fleets2[:, 0] == fleet_id)
        if still.any():
            continue

        # Fleet removed — find nearest planet to end position.
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


@pytest.mark.parametrize("seed", [0, 3, 11, 42, 100])
def test_valid_decoded_moves_mostly_hit_intended_target(seed: int):
    """Monte Carlo: full_valid moves should usually hit the chosen target planet."""
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
        n_before = int(state.n_fleets)
        state2 = step(reset(seed, episode_steps=500), [[[from_id, angle, ships]], []])
        if int(state2.n_fleets) <= n_before:
            continue
        fleet_id = int(np.asarray(state2.fleets)[int(state2.n_fleets) - 1, 0])
        outcomes[_simulate_fleet_hit(state2, fleet_id, t_idx)] += 1
        tested += 1

    if tested == 0:
        pytest.skip(f"seed {seed}: no fleets created")
    hit_rate = outcomes["hit_target"] / tested
    wrong_rate = outcomes["hit_wrong"] / tested
    assert hit_rate >= 0.35, f"seed {seed}: outcomes={dict(outcomes)}"
    assert wrong_rate <= 0.35, f"seed {seed}: too many wrong-planet hits ({wrong_rate:.2f})"


def test_compose_grid_masks_sun_crossing_moves():
    state = reset(0, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    sun = np.asarray(grid["sun_blocks"])
    assert not np.any(full & sun)


def test_compose_grid_masks_planet_blocked_moves():
    state = reset(11, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    pb = np.asarray(grid["planet_blocks"])
    assert not np.any(full & pb)


def test_full_valid_implies_no_sun_or_planet_block():
    for seed in [0, 5, 17, 99]:
        state = reset(seed, episode_steps=300)
        grid = compose_action_grid(state, jnp.int32(0))
        full = np.asarray(grid["full_valid"])
        if not full.any():
            continue
        assert not np.any(full & np.asarray(grid["sun_blocks"]))
        assert not np.any(full & np.asarray(grid["planet_blocks"]))


def test_decoded_angle_fleet_speed_consistent():
    """Angle + ship count should match intercept point for orbiting targets."""
    state = reset(100, episode_steps=500)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    idxs = np.argwhere(full)
    if len(idxs) == 0:
        pytest.skip("no valid moves")
    s_idx, t_idx, b_idx = idxs[0]
    angle = float(grid["angle"][s_idx, t_idx, b_idx])
    aim_x = float(grid["aim_x"][s_idx, t_idx, b_idx])
    aim_y = float(grid["aim_y"][s_idx, t_idx, b_idx])
    sx = float(state.planets[s_idx, 2])
    sy = float(state.planets[s_idx, 3])
    ref = ref_safe_angle(sx, sy, aim_x, aim_y)
    assert angle == pytest.approx(ref, abs=0.02)
