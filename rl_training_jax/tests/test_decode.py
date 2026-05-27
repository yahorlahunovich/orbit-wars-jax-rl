"""Tests for the JAX action decoder."""

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
from orbit_wars.constants import CENTER, SUN_RADIUS
from orbit_wars.decode import (
    BUCKET_COUNT,
    bucket_validity_mask,
    compose_action_grid,
    launch_angle,
    pack_action_row,
    path_crosses_sun,
    ship_counts_for_buckets,
)


def test_ship_counts_monotone_in_source():
    """Larger source ships should not decrease per-bucket ship counts."""
    sc_small = ship_counts_for_buckets(jnp.float32(50.0), jnp.float32(10.0))
    sc_big = ship_counts_for_buckets(jnp.float32(500.0), jnp.float32(10.0))
    assert np.all(np.asarray(sc_big) >= np.asarray(sc_small))


def test_ship_count_floor_and_minimum():
    """All buckets must produce integer-valued ship counts >= 1."""
    sc = ship_counts_for_buckets(jnp.float32(3.0), jnp.float32(2.0))
    arr = np.asarray(sc)
    assert arr.shape == (BUCKET_COUNT,)
    assert np.all(arr >= 1.0)
    assert np.all(arr == np.floor(arr))


def test_bucket_validity_respects_source_cap():
    sc = ship_counts_for_buckets(jnp.float32(5.0), jnp.float32(100.0))
    mask = bucket_validity_mask(sc, jnp.float32(5.0))
    arr = np.asarray(mask)
    # Capture buckets (5, 6) need target_ships+ ships -> 101, 102 -> too many.
    assert not arr[5]
    assert not arr[6]
    # Fractional buckets (0..4) with source=5 -> 1..5 ships -> all valid.
    assert np.all(arr[:5])
    assert arr[7]  # constant bucket = 4


def test_path_crosses_sun_diagonal():
    # Path crossing through the centre is blocked.
    blocked = path_crosses_sun(
        jnp.float32(10.0), jnp.float32(50.0),
        jnp.float32(90.0), jnp.float32(50.0),
    )
    assert bool(blocked)
    # Path tangent to board edge far from sun is fine.
    clear = path_crosses_sun(
        jnp.float32(10.0), jnp.float32(95.0),
        jnp.float32(90.0), jnp.float32(95.0),
    )
    assert not bool(clear)


def test_launch_angle_quadrants():
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0),
                           jnp.float32(1.0), jnp.float32(0.0)))
    assert np.isclose(a, 0.0)
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0),
                           jnp.float32(0.0), jnp.float32(1.0)))
    assert np.isclose(a, np.pi / 2)
    a = float(launch_angle(jnp.float32(0.0), jnp.float32(0.0),
                           jnp.float32(-1.0), jnp.float32(0.0)))
    assert np.isclose(a, np.pi)


def test_compose_action_grid_shapes():
    state = reset(0, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    assert grid["source_valid"].shape == (MAX_PLANETS,)
    assert grid["angle"].shape == (MAX_PLANETS, MAX_PLANETS)
    assert grid["sun_blocks"].shape == (MAX_PLANETS, MAX_PLANETS)
    assert grid["ship_counts"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)
    assert grid["bucket_valid"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)
    assert grid["full_valid"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)


def test_full_valid_implies_pair_valid_and_bucket_valid():
    state = reset(11, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    pair = np.asarray(grid["pair_valid"])
    bucket = np.asarray(grid["bucket_valid"])
    assert np.all(~full | (pair[..., None] & bucket))
    # No self-targeted move should ever be marked valid.
    diag = np.arange(MAX_PLANETS)
    assert not np.any(full[diag, diag, :])


def test_full_valid_implies_source_owned_by_player():
    """Only sources owned by `player` may have any valid action."""
    state = reset(3, episode_steps=200)
    grid0 = compose_action_grid(state, jnp.int32(0))
    grid1 = compose_action_grid(state, jnp.int32(1))
    fv0 = np.asarray(grid0["full_valid"]).any(axis=(1, 2))
    fv1 = np.asarray(grid1["full_valid"]).any(axis=(1, 2))
    src0 = np.asarray(grid0["source_valid"])
    src1 = np.asarray(grid1["source_valid"])
    # Implication: full_valid for any (target, bucket) -> source_valid.
    assert np.all(~fv0 | src0)
    assert np.all(~fv1 | src1)


def test_compose_action_grid_is_jittable():
    state = reset(7, episode_steps=200)

    @jax.jit
    def f(s):
        return compose_action_grid(s, jnp.int32(0))

    out = f(state)
    assert out["full_valid"].shape == (MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)


def test_decoded_move_executes_in_env():
    """End-to-end: pick a valid (source, target, bucket), pack the move, step
    the env, confirm a new fleet was created."""
    state = reset(0, episode_steps=200)
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    n_fleets_before = int(state.n_fleets)

    # Find any valid (s, t, b).
    idxs = np.argwhere(full)
    assert len(idxs) > 0, "no valid moves in this seed/state"
    s_idx, t_idx, b_idx = idxs[0]

    from_id = float(grid["from_ids"][s_idx])
    angle = float(grid["angle"][s_idx, t_idx])
    ships = int(grid["ship_counts"][s_idx, t_idx, b_idx])
    move = [[from_id, angle, ships]]
    state2 = step(state, [move, []])
    assert int(state2.n_fleets) == n_fleets_before + 1


def test_pack_action_row_invalid_returns_zero():
    row, mask = pack_action_row(
        jnp.float32(3.0), jnp.float32(1.5), jnp.float32(10.0), jnp.bool_(False)
    )
    assert np.all(np.asarray(row) == 0.0)
    assert float(mask) == 0.0
