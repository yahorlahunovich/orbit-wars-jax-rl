"""Compare learner vs heuristic in reference (Kaggle) env vs JAX env."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as tu
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import (
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLANET_FEATURE_DIM,
    compose_action_grid,
    encode_observation,
    reset,
)
from orbit_wars.constants import MAX_FLEETS, MAX_PLANETS
from orbit_wars.convert import state_to_observation_dict
from orbit_wars.heuristic_opponent import load_heuristic_agent
from orbit_wars.reference import episode_seed_from_env, reference_reset, reference_step
from orbit_wars.rollout import pack_padded_actions, sample_actions
from orbit_wars.step import step as jax_step
from policy import PlanetPolicy


def _rows_from_packed(actions, mask):
    a = np.asarray(actions)
    m = np.asarray(mask)
    return [
        [float(a[i, 0]), float(a[i, 1]), float(a[i, 2])]
        for i in range(len(m))
        if m[i] > 0
    ]


def _play_reference(seed: int, params, model, heuristic, rng):
    env, ref = reference_reset(seed, episode_steps=500)
    episode_seed = episode_seed_from_env(env)
    from orbit_wars.convert import observation_to_state

    state = observation_to_state(ref.observations[0], episode_seed=episode_seed, episode_steps=500)
    learner = seed % 2
    for _ in range(520):
        obs0 = state_to_observation_dict(state, player=0)
        obs1 = state_to_observation_dict(state, player=1)
        if ref.done or bool(state.done):
            break

        feats0 = {k: v[None] for k, v in encode_observation(state, 0).items()}
        feats1 = {k: v[None] for k, v in encode_observation(state, 1).items()}
        out0 = model.apply(params, **feats0)
        out1 = model.apply(params, **feats1)
        g0 = {k: v[None] for k, v in compose_action_grid(state, 0).items()}
        g1 = {k: v[None] for k, v in compose_action_grid(state, 1).items()}
        rng, k0, k1 = jr.split(rng, 3)
        s0 = sample_actions(k0, out0.target_logits, out0.bucket_logits, g0)
        s1 = sample_actions(k1, out1.target_logits, out1.bucket_logits, g1)
        a0, m0 = pack_padded_actions(s0["target_idx"], s0["bucket_idx"], s0["source_valid"], g0)
        a1, m1 = pack_padded_actions(s1["target_idx"], s1["bucket_idx"], s1["source_valid"], g1)
        rows0 = _rows_from_packed(a0[0], m0[0])
        rows1 = _rows_from_packed(a1[0], m1[0])
        h0 = heuristic(obs0)
        h1 = heuristic(obs1)
        if learner == 0:
            p0, p1 = rows0, h1
        else:
            p0, p1 = h0, rows1
        ref = reference_step(env, [p0, p1])
        state = jax_step(state, [p0, p1])

    lr = float(state.rewards[learner])
    hr = float(state.rewards[1 - learner])
    return lr, hr


def main() -> None:
    model = PlanetPolicy(
        planet_count=MAX_PLANETS,
        fleet_count=MAX_FLEETS,
        d_model=96,
        num_heads=4,
        num_layers=3,
        bucket_count=8,
    )
    params = model.init(
        jr.PRNGKey(1),
        planet_features=jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        planet_mask=jnp.ones((1, MAX_PLANETS), jnp.bool_),
    )
    heuristic = load_heuristic_agent()
    rng = jr.PRNGKey(0)
    w = l = 0
    for seed in range(8):
        rng, sub = jr.split(rng)
        lr, hr = _play_reference(seed, params, model, heuristic, sub)
        if lr > hr:
            w += 1
        elif lr < hr:
            l += 1
        print(f"seed={seed} learner={lr:+.0f} heur={hr:+.0f}")
    print(f"summary W-L: {w}-{l}")


if __name__ == "__main__":
    main()
