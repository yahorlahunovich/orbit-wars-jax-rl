"""Tests for PPO loss + GAE."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import (
    BUCKET_COUNT,
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_FLEETS,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
)
from policy import PlanetPolicy
from ppo import compute_gae, explained_variance, ppo_loss_fn


def test_gae_terminal_zero_advantage_for_zero_reward():
    rewards = jnp.zeros((2, 5), dtype=jnp.float32)
    values = jnp.zeros((2, 5), dtype=jnp.float32)
    dones = jnp.zeros((2, 5), dtype=jnp.bool_)
    next_value = jnp.zeros((2,), dtype=jnp.float32)
    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
    assert np.allclose(np.asarray(adv), 0.0)
    assert np.allclose(np.asarray(ret), 0.0)


def test_gae_terminal_propagates_reward():
    """A +1 reward at the last step should produce positive advantages back
    through time, with discounting."""
    rewards = jnp.zeros((1, 4), dtype=jnp.float32).at[0, 3].set(1.0)
    values = jnp.zeros((1, 4), dtype=jnp.float32)
    dones = jnp.zeros((1, 4), dtype=jnp.bool_).at[0, 3].set(True)
    next_value = jnp.zeros((1,), dtype=jnp.float32)
    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=1.0)
    adv = np.asarray(adv)
    # With gamma=lam=1 and zero values, advantage at each step = sum of future rewards = 1.
    assert np.allclose(adv[0, :], [1.0, 1.0, 1.0, 1.0])


def test_gae_done_resets_bootstrap():
    """An episode boundary in the middle should stop reward propagation."""
    rewards = jnp.array([[0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    values = jnp.zeros((1, 4), dtype=jnp.float32)
    dones = jnp.array([[False, True, False, False]], dtype=jnp.bool_)
    next_value = jnp.zeros((1,), dtype=jnp.float32)
    adv, _ = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=1.0)
    a = np.asarray(adv)
    assert np.isclose(a[0, 1], 1.0)
    assert np.isclose(a[0, 2], 0.0)
    assert np.isclose(a[0, 3], 0.0)


def test_explained_variance_constant_returns_zero():
    ev = explained_variance(jnp.ones((10,), jnp.float32), jnp.zeros((10,), jnp.float32))
    assert float(ev) == 0.0


def test_explained_variance_perfect_predictor():
    r = jnp.asarray(np.random.randn(50).astype(np.float32))
    ev = explained_variance(r, r)
    assert float(ev) > 0.99


def test_ppo_loss_runs_and_decreases_with_better_policy():
    """Smoke: build a minimal batch, take a grad step, check loss is finite
    and gradients are non-zero."""
    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(
        planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS,
        d_model=32, num_heads=4, num_layers=1, bucket_count=BUCKET_COUNT,
    )
    N = 4
    P = MAX_PLANETS
    example = {
        "planet_features": jnp.ones((N, P, PLANET_FEATURE_DIM), jnp.float32) * 0.1,
        "planet_mask": jnp.ones((N, P), jnp.bool_),
        "fleet_features": jnp.zeros((N, MAX_FLEETS, FLEET_FEATURE_DIM), jnp.float32),
        "fleet_mask": jnp.zeros((N, MAX_FLEETS), jnp.bool_),
        "global_features": jnp.zeros((N, GLOBAL_FEATURE_DIM), jnp.float32),
    }
    params = model.init(rng, **example)

    target_has_bucket = jnp.ones((N, P, P), jnp.bool_)
    bucket_valid = jnp.ones((N, P, P, BUCKET_COUNT), jnp.bool_)
    executed_mask = jnp.zeros((N, P), jnp.bool_).at[:, :3].set(True)
    target_idx = jnp.zeros((N, P), jnp.int32).at[:, :3].set(1)
    bucket_idx = jnp.zeros((N, P), jnp.int32)
    old_log_prob = jnp.zeros((N, P), jnp.float32)
    advantages = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)
    returns = jnp.array([0.5, -0.5, 0.25, -0.25], dtype=jnp.float32)

    batch = {
        **example,
        "target_has_bucket": target_has_bucket,
        "bucket_valid": bucket_valid,
        "executed_mask": executed_mask,
        "target_idx": target_idx,
        "bucket_idx": bucket_idx,
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "returns": returns,
    }

    def loss_fn(p):
        return ppo_loss_fn(p, model.apply, batch, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01)

    (l, metrics), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert jnp.isfinite(l)
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"{k}={v} not finite"
    # At least some grad leaves are nonzero.
    flat_g = jax.tree_util.tree_leaves(g)
    norms = [float(jnp.linalg.norm(x)) for x in flat_g]
    assert max(norms) > 0.0
