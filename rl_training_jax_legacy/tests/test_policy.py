"""Smoke tests for the Transformer policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from policy import PlanetPolicy, init_policy
from orbit_wars import (
    MAX_FLEETS,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    encode_observation,
    reset,
)


def _example_batch(batch: int = 2):
    return {
        "planet_features": jnp.zeros((batch, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((batch, MAX_PLANETS), jnp.bool_),
    }


def test_policy_forward_shapes():
    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS, bucket_count=8)
    example = _example_batch(batch=3)
    params = init_policy(rng, model, example)
    out = model.apply(params, **example)
    assert out.target_logits.shape == (3, MAX_PLANETS, MAX_PLANETS)
    assert out.bucket_logits.shape == (3, MAX_PLANETS, MAX_PLANETS, 8)
    assert out.value.shape == (3,)
    assert jnp.isfinite(out.target_logits).all()
    assert jnp.isfinite(out.bucket_logits).all()
    assert jnp.isfinite(out.value).all()


def test_policy_jit_and_backward():
    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS, bucket_count=8)
    example = _example_batch(batch=4)
    params = init_policy(rng, model, example)

    @jax.jit
    def loss_fn(p, batch):
        out = model.apply(p, **batch)
        return out.value.mean() + out.target_logits.mean() + out.bucket_logits.mean()

    g = jax.grad(loss_fn)(params, example)
    flat = tu.tree_leaves(g)
    assert flat, "no grads returned"
    for arr in flat:
        assert jnp.all(jnp.isfinite(arr))


def test_policy_consumes_real_features():
    """Pipe a real OrbitWarsState through the encoder + policy."""
    rng = jax.random.PRNGKey(42)
    state = reset(7, episode_steps=200)
    obs = encode_observation(state, jnp.int32(0))
    batch = {k: v[None, ...] for k, v in obs.items()}
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS)
    params = model.init(rng, **batch)
    out = model.apply(params, **batch)
    assert out.target_logits.shape == (1, MAX_PLANETS, MAX_PLANETS)
    assert out.value.shape == (1,)


def test_param_count_reasonable():
    """Param count should fit a Kaggle submission (≤ ~1.5M)."""
    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(
        planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS,
        d_model=96, num_heads=4, num_layers=3, bucket_count=8,
    )
    example = _example_batch(batch=2)
    params = init_policy(rng, model, example)
    total = sum(arr.size for arr in tu.tree_leaves(params))
    assert total < 1_500_000, f"too big: {total}"
    print(f"param count: {total}")


def test_padding_does_not_affect_active_outputs():
    """Toggling padding-row features should not change outputs for active rows
    (mask should be honored throughout attention)."""
    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS)
    example = _example_batch(batch=1)
    # Mark first 10 planets as active, the rest padded.
    planet_mask = jnp.zeros((1, MAX_PLANETS), jnp.bool_).at[:, :10].set(True)
    batch_a = {**example, "planet_mask": planet_mask}

    # Perturb only padding rows.
    planet_features_b = example["planet_features"].at[0, 50:, :].set(7.5)
    batch_b = {
        **batch_a,
        "planet_features": planet_features_b,
    }

    params = init_policy(rng, model, batch_a)
    out_a = model.apply(params, **batch_a)
    out_b = model.apply(params, **batch_b)

    # Active source rows (the first 10) should be identical between the two.
    assert np.allclose(
        np.asarray(out_a.target_logits[:, :10, :10]),
        np.asarray(out_b.target_logits[:, :10, :10]),
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(out_a.bucket_logits[:, :10, :10, :]),
        np.asarray(out_b.bucket_logits[:, :10, :10, :]),
        atol=1e-5,
    )
    assert np.allclose(np.asarray(out_a.value), np.asarray(out_b.value), atol=1e-5)
