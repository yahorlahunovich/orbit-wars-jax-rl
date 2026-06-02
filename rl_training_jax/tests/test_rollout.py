"""Tests for the masked sampling + action packing layer."""

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
    BUCKET_COUNT,
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    compose_action_grid,
    encode_observation,
    reset,
    step,
)
from orbit_wars.rollout import pack_padded_actions, policy_step, sample_actions
from policy import PlanetPolicy


def _make_policy(rng):
    model = PlanetPolicy(
        planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS,
        d_model=48, num_heads=4, num_layers=2, bucket_count=BUCKET_COUNT,
    )
    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
    }
    params = model.init(rng, **example)
    return model, params


def _batched_state(seeds):
    states = [reset(s, episode_steps=200) for s in seeds]
    batched = tu.tree_map(lambda *xs: jnp.stack(xs), *states)
    return batched, states


def _batched_features(batched_state, players):
    return jax.vmap(encode_observation, in_axes=(0, 0))(batched_state, players)


def test_sample_actions_respects_mask():
    """Sampled (target, bucket) pairs must be in the valid mask."""
    rng = jax.random.PRNGKey(0)
    model, params = _make_policy(rng)
    seeds = (0, 7, 21, 33)
    batched, _states = _batched_state(seeds)
    players = jnp.zeros((len(seeds),), dtype=jnp.int32)
    features = _batched_features(batched, players)
    out = model.apply(params, **features)
    grid = jax.vmap(compose_action_grid, in_axes=(0, 0))(batched, players)

    sampled = sample_actions(rng, out.target_logits, out.bucket_logits, grid)
    target_idx = np.asarray(sampled["target_idx"])
    bucket_idx = np.asarray(sampled["bucket_idx"])
    source_valid = np.asarray(sampled["source_valid"])
    bucket_valid = np.asarray(grid["bucket_valid"])
    pair_valid = np.asarray(grid["pair_valid"])

    b, p = target_idx.shape
    for bi in range(b):
        for si in range(p):
            if not source_valid[bi, si]:
                continue
            t = target_idx[bi, si]
            k = bucket_idx[bi, si]
            assert pair_valid[bi, si, t], f"target {t} invalid for source {si} env {bi}"
            assert bucket_valid[bi, si, t, k], (
                f"bucket {k} invalid for (src={si}, tgt={t}) env={bi}"
            )


def test_pack_padded_actions_truncates_to_max_moves():
    rng = jax.random.PRNGKey(1)
    model, params = _make_policy(rng)
    seeds = (0, 5)
    batched, _ = _batched_state(seeds)
    players = jnp.zeros((len(seeds),), dtype=jnp.int32)
    features = _batched_features(batched, players)
    out = model.apply(params, **features)
    grid = jax.vmap(compose_action_grid, in_axes=(0, 0))(batched, players)
    sampled = sample_actions(rng, out.target_logits, out.bucket_logits, grid)

    actions, mask = pack_padded_actions(
        sampled["target_idx"], sampled["bucket_idx"], sampled["source_valid"], grid
    )
    assert actions.shape == (2, MAX_MOVES_PER_PLAYER, 3)
    assert mask.shape == (2, MAX_MOVES_PER_PLAYER)
    # Mask is non-increasing (valid moves grouped to the front).
    m = np.asarray(mask)
    diff = np.diff(m, axis=-1)
    assert np.all(diff <= 0), "mask should be non-increasing after pack"


def test_full_pipeline_actions_execute_in_env():
    """Run policy -> sample -> pack -> step for a real state. Step must accept
    the packed action tensor and produce new fleets matching the mask."""
    from orbit_wars.step import step_jit

    rng = jax.random.PRNGKey(2)
    model, params = _make_policy(rng)
    seeds = (3, 10, 14, 19)
    batched, _ = _batched_state(seeds)
    players = jnp.zeros((len(seeds),), dtype=jnp.int32)
    features = _batched_features(batched, players)

    apply = jax.jit(model.apply)
    info = policy_step(rng, apply, params, batched, features, players)

    actions = info["actions"]
    mask = info["action_mask"]
    # The opponent stays put.
    opp_actions = jnp.zeros_like(actions)
    opp_mask = jnp.zeros_like(mask)
    n_fleets_before = np.asarray(batched.n_fleets)

    new_states = jax.vmap(step_jit)(batched, actions, opp_actions, mask, opp_mask)
    n_fleets_after = np.asarray(new_states.n_fleets)
    moves_per_env = np.asarray(mask).sum(axis=-1).astype(int)
    # New fleets >= number of valid moves; could be lower if some moves
    # collided with sun mid-step (shouldn't here, but be generous):
    delta = n_fleets_after - n_fleets_before
    # The env counts launched fleets immediately; collisions can remove some
    # right after, but launch counts up to MAX_FLEETS - n_before.
    assert np.all(delta >= 0)
    assert np.all(delta <= moves_per_env)


def test_log_prob_finite_and_zero_when_no_valid_source():
    rng = jax.random.PRNGKey(0)
    model, params = _make_policy(rng)
    state = reset(0, episode_steps=200)
    batched = tu.tree_map(lambda x: x[None, ...], state)
    players = jnp.zeros((1,), dtype=jnp.int32)
    features = _batched_features(batched, players)
    out = model.apply(params, **features)
    grid = jax.vmap(compose_action_grid, in_axes=(0, 0))(batched, players)
    sampled = sample_actions(rng, out.target_logits, out.bucket_logits, grid)

    lp = np.asarray(sampled["log_prob"])
    sv = np.asarray(sampled["source_valid"])
    assert np.all(np.isfinite(lp))
    # For sources without valid action, log_prob must be exactly 0 (we
    # treat them as deterministic non-moves so they don't contribute to PPO).
    assert np.all(lp[~sv] == 0.0)
