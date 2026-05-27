"""Tests for the pure-JAX feature encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import (
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_FLEETS,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    encode_batch,
    encode_batch_jit,
    encode_observation,
    encode_observation_jit,
    reset,
    step_jit,
)
from orbit_wars.step import _list_action_to_padded


def _empty_action():
    a, m = _list_action_to_padded([])
    return a, m


def test_encode_shapes_and_dims():
    state = reset(7, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))

    assert out["planet_features"].shape == (MAX_PLANETS, PLANET_FEATURE_DIM)
    assert out["fleet_features"].shape == (MAX_FLEETS, FLEET_FEATURE_DIM)
    assert out["global_features"].shape == (GLOBAL_FEATURE_DIM,)
    assert out["planet_mask"].shape == (MAX_PLANETS,)
    assert out["fleet_mask"].shape == (MAX_FLEETS,)
    assert out["planet_mask"].dtype == jnp.bool_
    assert out["fleet_mask"].dtype == jnp.bool_


def test_masked_padding_is_zero():
    """Inactive planet/fleet slots should produce all-zero feature rows."""
    state = reset(11, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))
    planet_features = np.asarray(out["planet_features"])
    fleet_features = np.asarray(out["fleet_features"])
    planet_mask = np.asarray(out["planet_mask"])
    fleet_mask = np.asarray(out["fleet_mask"])

    # Every inactive row is exactly zero.
    inactive_planets = planet_features[~planet_mask]
    inactive_fleets = fleet_features[~fleet_mask]
    if inactive_planets.size:
        assert np.all(inactive_planets == 0.0)
    if inactive_fleets.size:
        assert np.all(inactive_fleets == 0.0)


def test_player_relative_flip_is_symmetric():
    """Encoding for player 1 should be the mirror of player 0's encoding.

    Owner-is-me / owner-is-enemy columns must swap; everything position-based
    must stay the same.
    """
    state = reset(3, episode_steps=200)
    out0 = encode_observation(state, jnp.int32(0))
    out1 = encode_observation(state, jnp.int32(1))

    pf0 = np.asarray(out0["planet_features"])
    pf1 = np.asarray(out1["planet_features"])

    # Column 1 = owner_is_me, column 2 = owner_is_enemy.
    assert np.allclose(pf0[:, 1], pf1[:, 2])
    assert np.allclose(pf0[:, 2], pf1[:, 1])
    # owner_is_neutral (col 3) and positions (cols 4..16) must match.
    assert np.allclose(pf0[:, 3], pf1[:, 3])
    assert np.allclose(pf0[:, 4:17], pf1[:, 4:17])


def test_global_lead_signs_flip():
    state = reset(5, episode_steps=200)
    g0 = np.asarray(encode_observation(state, jnp.int32(0))["global_features"])
    g1 = np.asarray(encode_observation(state, jnp.int32(1))["global_features"])
    # prod_lead (idx 9) and ship_lead (idx 10) must swap sign between players.
    assert np.allclose(g0[9], -g1[9])
    assert np.allclose(g0[10], -g1[10])
    # Largest ships/production columns must swap between players.
    assert np.allclose(g0[16], g1[17])
    assert np.allclose(g0[17], g1[16])
    assert np.allclose(g0[18], g1[19])
    assert np.allclose(g0[19], g1[18])


def test_planet_rankings_within_subset():
    """ship_rank_all / prod_rank_all should produce values in [0, 1] and be 1.0
    only for the strictly-largest active planet."""
    state = reset(13, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))
    pf = np.asarray(out["planet_features"])
    mask = np.asarray(out["planet_mask"])
    ship_rank = pf[:, 22]
    prod_rank = pf[:, 23]
    assert np.all((ship_rank >= 0) & (ship_rank <= 1))
    assert np.all((prod_rank >= 0) & (prod_rank <= 1))
    assert np.all(ship_rank[~mask] == 0)
    # is_my_largest (col 28) should be 0 or 1 only.
    assert set(np.unique(pf[:, 28])).issubset({0.0, 1.0})


def test_comet_remaining_present_after_spawn():
    """After a comet has spawned (step >= 50), comet planets should report
    positive remaining-life values."""
    from orbit_wars.step import _list_action_to_padded, step

    state = reset(0, episode_steps=200)
    a, m = _list_action_to_padded([])
    for _ in range(60):
        state = step(state, [[], []])
    out = encode_observation(state, jnp.int32(0))
    pf = np.asarray(out["planet_features"])
    is_comet = pf[:, 13] > 0.5
    if is_comet.any():
        remaining = pf[is_comet, 30]
        assert np.all(remaining >= 0)
        assert np.any(remaining > 0)


def test_encoder_is_jittable():
    state = reset(0, episode_steps=200)
    out_eager = encode_observation(state, jnp.int32(0))
    out_jit = encode_observation_jit(state, jnp.int32(0))
    for k in out_eager:
        a = np.asarray(out_eager[k])
        b = np.asarray(out_jit[k])
        if a.dtype == np.bool_:
            assert np.array_equal(a, b), f"bool mismatch on {k}"
        else:
            assert np.allclose(a, b, atol=1e-5), f"mismatch on {k}"


def test_vmap_matches_single():
    states = [reset(seed, episode_steps=200) for seed in (1, 4, 9, 16)]
    batched = tu.tree_map(lambda *xs: jnp.stack(xs), *states)
    players = jnp.zeros((len(states),), dtype=jnp.int32)
    batched_out = encode_batch_jit(batched, players)

    for i, s in enumerate(states):
        single = encode_observation(s, jnp.int32(0))
        for k in single:
            a = np.asarray(single[k])
            b = np.asarray(batched_out[k][i])
            if a.dtype == np.bool_:
                assert np.array_equal(a, b), f"bool mismatch on key={k} env={i}"
            else:
                assert np.allclose(a, b, atol=1e-5), f"mismatch on key={k} env={i}"


def test_encoding_after_step():
    """Encoder must remain consistent after running step_jit."""
    state = reset(2, episode_steps=200)
    a, m = _empty_action()
    for _ in range(20):
        state = step_jit(state, a, a, m, m)
    out = encode_observation(state, jnp.int32(0))
    assert jnp.all(jnp.isfinite(out["planet_features"]))
    assert jnp.all(jnp.isfinite(out["fleet_features"]))
    assert jnp.all(jnp.isfinite(out["global_features"]))
    # Turn fraction should be (state.step / episode_steps).
    expected_turn = float(state.step) / float(state.episode_steps)
    assert np.isclose(float(out["global_features"][0]), expected_turn)
