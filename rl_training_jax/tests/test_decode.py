"""Tests for action decoding and ship count logic."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import MAX_PLANETS, reset, step
from orbit_wars.decode import (
    BUCKET_COUNT,
    compose_action_grid,
    compose_full_grid,
    launch_angle,
    pack_action_row,
    path_crosses_sun,
    ship_counts_for_buckets,
)


def test_ship_counts_monotone_in_source():
    """Larger source ships should not decrease per-bucket ship counts."""
    sc_small = ship_counts_for_buckets(jnp.float32(50.0), jnp.float32(10.0), jnp.float32(0.0), jnp.float32(0.0))
    sc_large = ship_counts_for_buckets(jnp.float32(100.0), jnp.float32(10.0), jnp.float32(0.0), jnp.float32(0.0))
    # All buckets should be >= small version
    assert np.all(np.asarray(sc_large) >= np.asarray(sc_small))


def test_ship_count_floor_and_minimum():
    """All buckets must produce integer-valued ship counts >= 1."""
    sc = ship_counts_for_buckets(jnp.float32(20.0), jnp.float32(2.0), jnp.float32(0.0), jnp.float32(0.0))
    sca = np.asarray(sc)
    assert np.all(sca == np.floor(sca))
    assert np.all(sca >= 1.0)


def test_bucket_validity_respects_source_cap():
    sc = ship_counts_for_buckets(jnp.float32(5.0), jnp.float32(100.0), jnp.float32(0.0), jnp.float32(0.0))
    sca = np.asarray(sc)
    # Buckets calculated based on needed ships (100+) should be capped at 5
    assert np.all(sca <= 5.0)


def test_launch_angle_quadrants():
    # 0 deg (east)
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0), jnp.float32(10.0), jnp.float32(0.0)))
    assert a == pytest.approx(0.0)
    # 90 deg (north)
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(10.0)))
    assert a == pytest.approx(np.pi / 2)
    # 180 deg (west)
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0), jnp.float32(-10.0), jnp.float32(0.0)))
    assert a == pytest.approx(np.pi)


def test_path_crosses_sun_diagonal():
    # SUN at (50, 50), radius 10.
    # Path (0, 0) -> (100, 100) passes through (50, 50).
    blocked = path_crosses_sun(
        jnp.float32(0.0), jnp.float32(0.0), jnp.float32(100.0), jnp.float32(100.0), margin=0.0
    )
    assert bool(blocked)

    # Path (0, 0) -> (10, 100) stays in top left.
    clear = path_crosses_sun(
        jnp.float32(0.0), jnp.float32(0.0), jnp.float32(10.0), jnp.float32(100.0), margin=0.0
    )
    assert not bool(clear)


def test_compose_action_grid_shapes():
    state = reset(0, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    assert grid["source_valid"].shape == (MAX_PLANETS,)
    assert grid["angle"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)
    # sun_blocks, planet_blocks, pair_valid, full_valid are present in compose_full_grid
    assert grid["sun_blocks"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)
    assert grid["full_valid"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)


def test_full_valid_implies_pair_valid_and_bucket_valid():
    state = reset(11, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    pair = np.asarray(grid["pair_valid"])
    bucket = np.asarray(grid["bucket_valid"])
    
    # 3D expand pair for elementwise comparison
    pair_3d = np.repeat(pair[:, :, None], BUCKET_COUNT, axis=-1)
    
    # full => pair AND bucket
    assert np.all(full <= pair_3d)
    assert np.all(full <= bucket)


def test_full_valid_implies_source_owned_by_player():
    """Only sources owned by `player` may have any valid action."""
    state = reset(3, episode_steps=200)
    grid0 = compose_action_grid(state, jnp.int32(0))
    grid1 = compose_action_grid(state, jnp.int32(1))

    f0 = np.asarray(grid0["full_valid"])
    f1 = np.asarray(grid1["full_valid"])

    planets = np.asarray(state.planets)
    owners = planets[:, 1]
    
    # Check p0 moves
    for i in range(MAX_PLANETS):
        if owners[i] != 0:
            assert np.sum(f0[i]) == 0
            
    # Check p1 moves
    for i in range(MAX_PLANETS):
        if owners[i] != 1:
            assert np.sum(f1[i]) == 0


def test_compose_action_grid_is_jittable():
    state = reset(7, episode_steps=200)

    @jax.jit
    def f(s):
        return compose_action_grid(s, jnp.int32(0))

    out = f(state)
    assert "full_valid" in out


def test_decoded_move_executes_in_env():
    """End-to-end: pick a valid (source, target, bucket), pack the move, step
    the env, confirm a new fleet was created."""
    state = reset(0, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    
    full = np.asarray(grid["full_valid"])
    if not np.any(full):
        pytest.skip("No valid moves found for state 0, p0.")

    # Find the first valid (s, t, b)
    s, t, b = np.argwhere(full)[0]
    
    # Get angle and ships for this specific bucket
    angle = float(grid["angle"][s, t, b])
    ships = float(grid["ship_counts"][s, t, b])
    from_id = float(grid["from_ids"][s])
    
    # Manually build a (1, 3) move and pack it
    # We can use pack_action_row for scalar if we are careful
    # But wait, step() takes a list of lists.
    action_list = [[from_id, angle, ships]]
    
    n_fleets_before = int(state.n_fleets)
    next_state = step(state, [action_list, []])
    n_fleets_after = int(next_state.n_fleets)
    
    assert n_fleets_after == n_fleets_before + 1
