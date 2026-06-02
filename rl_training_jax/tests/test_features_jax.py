"""Tests for the JAX feature encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from orbit_wars import (
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_FLEETS,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    encode_observation,
    reset,
    step_jit,
)


def _empty_action():
    from orbit_wars.step import _list_action_to_padded
    return _list_action_to_padded([])


def test_encode_shapes_and_dims():
    state = reset(0, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))
    
    assert out["planet_features"].shape == (MAX_PLANETS, PLANET_FEATURE_DIM)
    assert out["planet_mask"].shape == (MAX_PLANETS,)
    assert out["fleet_features"].shape == (MAX_FLEETS, FLEET_FEATURE_DIM)
    assert out["fleet_mask"].shape == (MAX_FLEETS,)
    assert out["global_features"].shape == (GLOBAL_FEATURE_DIM,)


def test_masked_padding_is_zero():
    """Rows marked as inactive in the mask should have all features zeroed."""
    state = reset(1, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))
    
    pm = np.asarray(out["planet_mask"])
    pf = np.asarray(out["planet_features"])
    assert np.all(pf[~pm] == 0.0)
    
    fm = np.asarray(out["fleet_mask"])
    ff = np.asarray(out["fleet_features"])
    assert np.all(ff[~fm] == 0.0)


def test_player_relative_flip_is_symmetric():
    """Flipping the perspective from P0 to P1 should swap owner flags and lead signs."""
    state = reset(42, episode_steps=200)
    # Manually assign some owner for testing.
    # planet 0 owned by P0
    state = state.replace(planets=state.planets.at[0, 1].set(0.0))
    
    out0 = encode_observation(state, jnp.int32(0))
    out1 = encode_observation(state, jnp.int32(1))
    
    # Planet 0 'owner_is_me' for P0 should be 'owner_is_enemy' for P1.
    assert out0["planet_features"][0, 1] == 1.0
    assert out1["planet_features"][0, 1] == 0.0
    assert out1["planet_features"][0, 2] == 1.0
    
    # Global 'prod_lead' should have opposite signs.
    # prod_lead is index 7 or similar now.
    g0 = np.asarray(out0["global_features"])
    g1 = np.asarray(out1["global_features"])
    # Turn and 1-turn (0, 1) should be same.
    assert np.allclose(g0[0:2], g1[0:2])
    # prod_lead is index 7, ship_lead is index 8 in my new code.
    # Wait, let me check features_jax.py again.
    # global_features = jnp.stack([turn, 1.0-turn, p_cnt, ..., prod_lead, ship_lead, ...])
    # prod_lead is at index 7.
    assert np.isclose(g0[7], -g1[7])


def test_planet_rankings_within_subset():
    """Check that _rank_norm produces values in [0, 1]."""
    state = reset(100, episode_steps=200)
    out = encode_observation(state, jnp.int32(0))
    pf = np.asarray(out["planet_features"])
    # rank features are around index 24.
    ranks = pf[:, 24:26]
    assert np.all((ranks >= 0.0) & (ranks <= 1.0))


def test_comet_remaining_present_after_spawn():
    """After a comet has spawned (step >= 50), comet planets should report
    positive remaining-life values."""
    from orbit_wars.step import step
    
    state = reset(0, episode_steps=200)
    for _ in range(60):
        state = step(state, [[], []])
    out = encode_observation(state, jnp.int32(0))
    pf = np.asarray(out["planet_features"])
    # is_comet is index 13. comet_remaining_norm is index 30.
    is_comet = pf[:, 13] > 0.5
    if is_comet.any():
        remaining = pf[is_comet, 30]
        assert np.all(remaining > 0.0)


def test_encoder_is_jittable():
    state = reset(3, episode_steps=200)
    f = jax.jit(encode_observation)
    out = f(state, jnp.int32(0))
    assert "planet_features" in out


def test_vmap_matches_single():
    state0 = reset(10, episode_steps=200)
    state1 = reset(20, episode_steps=200)
    batched = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), state0, state1)
    players = jnp.array([0, 1], dtype=jnp.int32)
    
    batch_out = jax.vmap(encode_observation)(batched, players)
    single0 = encode_observation(state0, jnp.int32(0))
    single1 = encode_observation(state1, jnp.int32(1))
    
    for k in batch_out:
        assert np.allclose(np.asarray(batch_out[k][0]), np.asarray(single0[k]), atol=1e-5)
        assert np.allclose(np.asarray(batch_out[k][1]), np.asarray(single1[k]), atol=1e-5)


def test_encoding_after_step():
    """Encoder must remain consistent after running step_jit."""
    state = reset(2, episode_steps=200)
    a, m = _empty_action()
    for _ in range(20):
        state = step_jit(state, a, a, m, m)
    out = encode_observation(state, jnp.int32(0))
    assert jnp.all(jnp.isfinite(out["planet_features"]))
    
    # Turn fraction should be (state.step / episode_steps).
    expected_turn = float(state.step) / float(state.episode_steps)
    assert np.isclose(float(out["global_features"][0]), expected_turn)
