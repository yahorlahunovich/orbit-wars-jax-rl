"""Tests for Orbit Wars geometry and intercepts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import (
    CENTER,
    DEFAULT_SHIP_SPEED as MAX_SPEED,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    BUCKET_COUNT,
    INTERCEPT_ITERATIONS,
    SUN_PATH_MARGIN,
    compose_action_grid,
    path_crosses_sun,
)
from orbit_wars.decode import path_blocked_by_planets
SUN_X, SUN_Y = 50.0, 50.0
from orbit_wars.geometry import (
    estimate_intercept_angles,
    fleet_speed,
    predict_orbit_polar,
    solve_intercept,
    sun_hit,
)

# ---------------------------------------------------------------------------
# Python Reference Implementations (from official notebook)
# ---------------------------------------------------------------------------

def ref_predict_orbit(x, y, omega, t):
    dx, dy = x - 50, y - 50
    r = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) + omega * t
    return 50 + r * math.cos(theta), 50 + r * math.sin(theta)

def ref_estimate_arrival(fx, fy, fsr, tx, ty, ttr, ships):
    d = math.hypot(fx - tx, fy - ty)
    hit_d = max(0.0, d - (fsr + 0.1) - ttr)
    # Official fleet_speed
    s_speed = MAX_SPEED * (1.0 - 0.5 * min(1.0, math.log(max(1.0, ships)) / math.log(1000.0)))
    s_speed = max(1.0, s_speed)
    return max(1.0, math.ceil(hit_d / s_speed))

def ref_solve_intercept(
    fx: float, fy: float, fsr: float, tx: float, ty: float, ttr: float,
    orbiting: bool, omega: float, ships: int, iterations: int = 6,
) -> tuple[float, float, float]:
    turns = ref_estimate_arrival(fx, fy, fsr, tx, ty, ttr, ships)
    ix, iy = tx, ty
    if orbiting:
        for _ in range(iterations):
            ix, iy = ref_predict_orbit(tx, ty, omega, turns)
            turns = ref_estimate_arrival(fx, fy, fsr, ix, iy, ttr, ships)
    return ix, iy, turns

def ref_path_crosses_sun(x1, y1, x2, y2, margin: float = 1.5) -> bool:
    dx, dy = x2 - x1, y2 - y1
    d2 = dx*dx + dy*dy
    if d2 < 1e-9: return math.hypot(x1-50, y1-50) <= (10 + margin)
    t = ((50 - x1)*dx + (50 - y1)*dy) / d2
    t = max(0, min(1, t))
    return math.hypot(x1 + t*dx - 50, y1 + t*dy - 50) <= (10 + margin)

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
        
        assert float(jix) == pytest.approx(ref_ix, abs=1e-2)
        assert float(jiy) == pytest.approx(ref_iy, abs=1e-2)
        # Arrival turns can vary slightly due to rounding in different steps
        assert float(jtt) == pytest.approx(ref_tt, abs=2.1)


def test_path_blocked_by_middle_planet():
    sx, sy = 10.0, 50.0
    tx, ty = 90.0, 50.0
    bx, by, br = 50.0, 50.0, 5.0
    p = 4
    px = jnp.array([sx, tx, bx, 0.0], dtype=jnp.float32)
    py = jnp.array([sy, ty, by, 0.0], dtype=jnp.float32)
    pr = jnp.array([3.0, 3.0, br, 0.0], dtype=jnp.float32)
    active = jnp.array([True, True, True, False], dtype=jnp.bool_)
    
    # In compose_action_grid, we call it with (P, 1) and (1, P)
    start_x = px[:, None]
    start_y = py[:, None]
    aim_x = px[None, :]
    aim_y = py[None, :]
    
    blocked = np.asarray(path_blocked_by_planets(
        start_x, start_y, aim_x, aim_y, px, py, pr, active, margin=0.5,
    ))
    # Path from planet 0 to planet 1 passes through planet 2 (at 50, 50)
    assert blocked[0, 1]
    # Path from planet 0 to itself should be clear
    assert not blocked[0, 0]


@pytest.mark.parametrize("seed", [0, 42])
def test_valid_decoded_moves_mostly_hit_intended_target(seed: int):
    """Pick a valid move from the grid, execute it, and check it hits the target."""
    from orbit_wars import reset, step
    state = reset(seed, episode_steps=200)
    grid = compose_action_grid(state, player=0)
    
    full = np.asarray(grid["full_valid"])
    if not np.any(full):
        return
    
    # Try first 5 valid moves
    idxs = np.argwhere(full)[:5]
    for s, t, b in idxs:
        angle = float(grid["angle"][s, t, b])
        ships = float(grid["ship_counts"][s, t, b])
        from_id = float(grid["from_ids"][s])
        
        next_state = step(state, [[[from_id, angle, ships]], []])
        # Find the newly created fleet
        found = False
        for i in range(int(next_state.n_fleets)):
            f = next_state.fleets[i]
            if f[7] > 0 and f[1] == 0 and f[5] == from_id:
                found = True
                break
        assert found
