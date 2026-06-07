"""Diagnose training vs heuristic: move counts and win rate in JAX env."""

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
from orbit_wars.heuristic_opponent import batched_heuristic_actions, load_heuristic_agent
from orbit_wars.rollout import pack_padded_actions, sample_actions
from orbit_wars.step import step_jit
from policy import PlanetPolicy
from train_ppo import load_config, rollout_step_vs_heuristic_factory


def main() -> None:
    cfg = load_config(ROOT / "configs/transformer_curriculum.yaml")
    num_envs = 8
    episode_steps = cfg.episode_steps

    rng = jr.PRNGKey(0)
    states_list = []
    lp_np = np.zeros(num_envs, dtype=np.int32)
    r = random.Random(0)
    for i in range(num_envs):
        states_list.append(reset(i, episode_steps=episode_steps))
        lp_np[i] = r.randint(0, 1)
    states = tu.tree_map(lambda *xs: jnp.stack(xs), *states_list)
    learner_players = jnp.asarray(lp_np)

    model = PlanetPolicy(
        planet_count=MAX_PLANETS,
        fleet_count=MAX_FLEETS,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        bucket_count=cfg.bucket_count,
    )
    params = model.init(
        jr.PRNGKey(1),
        planet_features=jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        planet_mask=jnp.ones((1, MAX_PLANETS), jnp.bool_),
    )
    heuristic = load_heuristic_agent()
    step_one = rollout_step_vs_heuristic_factory(model)

    heur_move_steps = 0
    learner_w = learner_l = 0
    finished = 0
    total_steps = 0

    for _ in range(600):
        rng, sub = jr.split(rng)
        states, rec, rng = step_one(states, params, sub, learner_players, heuristic)
        total_steps += 1

        lp = np.asarray(learner_players)
        opp = 1 - lp
        _, hm0, _, hm1 = batched_heuristic_actions(states, opp, heuristic)
        hm0_np = np.asarray(hm0)
        hm1_np = np.asarray(hm1)
        heur_moves = float(np.sum(np.where(lp == 0, hm1_np.sum(axis=1), hm0_np.sum(axis=1))))
        if heur_moves > 0:
            heur_move_steps += 1

        done_np = np.asarray(rec["done"])
        reward_np = np.asarray(rec["reward"])
        if done_np.any():
            for i in range(num_envs):
                if done_np[i]:
                    r = float(reward_np[i])
                    if r > 0:
                        learner_w += 1
                    elif r < 0:
                        learner_l += 1
                    finished += 1
            break

    print(f"envs={num_envs} episode_steps={episode_steps}")
    print(f"finished={finished} learner W-L={learner_w}-{learner_l}")
    print(f"heuristic moved on {heur_move_steps}/{total_steps} steps")
    print(f"game ended at env step {int(np.max(np.asarray(states.step)))}")


if __name__ == "__main__":
    main()
