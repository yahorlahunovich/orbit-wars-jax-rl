"""Step-by-step parity tests against reference Kaggle env."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "rl_training"))

from orbit_wars.convert import observation_to_state, state_to_observation_dict
from orbit_wars.reference import episode_seed_from_env, reference_reset, reference_step
from orbit_wars.reset import reset
from orbit_wars.step import step


def _planets_almost_equal(ref_obs, jax_obs, *, tol: float = 0.05) -> bool:
    def rows(obs):
        out = []
        planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        for p in planets:
            out.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), int(p[5])))
        return sorted(out)

    a, b = rows(ref_obs), rows(jax_obs)
    if len(a) != len(b):
        return False
    for pa, pb in zip(a, b):
        if pa[0] != pb[0] or pa[1] != pb[1] or pa[4] != pb[4]:
            return False
        if abs(pa[2] - pb[2]) > tol or abs(pa[3] - pb[3]) > tol:
            return False
    return True


def _fleets_almost_equal(ref_obs, jax_obs, *, tol: float = 0.05) -> bool:
    def rows(obs):
        out = []
        fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
        for f in fleets:
            out.append((int(f[0]), int(f[1]), float(f[2]), float(f[3]), int(f[6])))
        return sorted(out)

    a, b = rows(ref_obs), rows(jax_obs)
    if len(a) != len(b):
        return False
    for fa, fb in zip(a, b):
        if fa[0] != fb[0] or fa[1] != fb[1] or fa[4] != fb[4]:
            return False
        if abs(fa[2] - fb[2]) > tol or abs(fa[3] - fb[3]) > tol:
            return False
    return True


@pytest.mark.parametrize("seed", [0, 1, 7, 21])
def test_reset_matches_reference(seed: int):
    env, ref = reference_reset(seed, episode_steps=120)
    episode_seed = episode_seed_from_env(env)
    jax_state = observation_to_state(
        ref.observations[0],
        episode_seed=episode_seed,
        episode_steps=120,
    )
    ref_obs = ref.observations[0]
    jax_obs = state_to_observation_dict(jax_state, player=0)
    assert _planets_almost_equal(ref_obs, jax_obs), f"seed={seed} planets mismatch after reset"


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_noop_rollout_parity_short(seed: int):
    env, ref = reference_reset(seed, episode_steps=80)
    episode_seed = episode_seed_from_env(env)
    state = observation_to_state(ref.observations[0], episode_seed=episode_seed, episode_steps=80)
    empty: list[list[float | int]] = []
    for step_idx in range(25):
        ref = reference_step(env, [empty, empty])
        state = step(state, [empty, empty])
        if ref.done or bool(state.done):
            break
        ref_obs = ref.observations[0]
        if not isinstance(ref_obs, dict):
            ref_obs = {
                "planets": ref_obs.planets,
                "fleets": ref_obs.fleets,
                "step": getattr(ref_obs, "step", step_idx + 1),
            }
        jax_obs = state_to_observation_dict(state, player=0)
        assert _planets_almost_equal(ref_obs, jax_obs), f"seed={seed} step={step_idx} planets"
        assert _fleets_almost_equal(ref_obs, jax_obs), f"seed={seed} step={step_idx} fleets"


def test_reset_helper():
    state = reset(42, episode_steps=100)
    assert int(state.step) == 1
    assert int(state.n_planets) > 0


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_noop_rollout_parity_full(seed: int):
    """Run a full episode of noop play to exercise comet spawn AND expiry."""
    episode = 500
    env, ref = reference_reset(seed, episode_steps=episode)
    episode_seed = episode_seed_from_env(env)
    state = observation_to_state(ref.observations[0], episode_seed=episode_seed, episode_steps=episode)
    empty: list[list[float | int]] = []
    for step_idx in range(episode - 5):
        ref = reference_step(env, [empty, empty])
        state = step(state, [empty, empty])
        if ref.done or bool(state.done):
            assert ref.done == bool(state.done), f"seed={seed} done mismatch at step {step_idx}"
            break
        ref_obs = ref.observations[0]
        jax_obs = state_to_observation_dict(state, player=0)
        assert _planets_almost_equal(ref_obs, jax_obs), f"seed={seed} step={step_idx} planets"
        assert _fleets_almost_equal(ref_obs, jax_obs), f"seed={seed} step={step_idx} fleets"


def test_batched_step_matches_single():
    """`batched_step` (vmap over step_jit) must match per-env step_jit results."""
    import jax
    import jax.numpy as jnp
    import jax.tree_util as tu

    from orbit_wars.step import batched_step, step_jit, _list_action_to_padded

    states = [reset(seed, episode_steps=200) for seed in (3, 11, 19, 25)]
    batched = tu.tree_map(lambda *xs: jnp.stack(xs), *states)
    a, m = _list_action_to_padded([])
    a_b = jnp.broadcast_to(a, (len(states), *a.shape))
    m_b = jnp.broadcast_to(m, (len(states), *m.shape))

    # Step both paths a few times and compare planets / fleets.
    for _ in range(5):
        batched = batched_step(batched, a_b, a_b, m_b, m_b)
        states = [step_jit(s, a, a, m, m) for s in states]

    bp = np.asarray(batched.planets)
    bf = np.asarray(batched.fleets)
    for i, s in enumerate(states):
        sp = np.asarray(s.planets)
        sf = np.asarray(s.fleets)
        assert np.allclose(bp[i], sp, atol=1e-4), f"planets mismatch env={i}"
        assert np.allclose(bf[i], sf, atol=1e-4), f"fleets mismatch env={i}"
