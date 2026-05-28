"""Benchmark JAX rollout throughput (env steps/sec) without PPO updates."""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars import MAX_FLEETS, MAX_PLANETS, compose_action_grid
from orbit_wars.heuristic_opponent import load_heuristic_agent
from policy import PlanetPolicy
from train_ppo import (
    TrainConfig,
    init_policy_params,
    make_initial_states,
    maybe_spawn_comets_host,
    reset_done_envs,
    rollout_step_selfplay_factory,
    rollout_step_vs_heuristic_factory,
    load_config,
)


def _build_model(cfg: TrainConfig):
    model = PlanetPolicy(
        planet_count=MAX_PLANETS,
        fleet_count=MAX_FLEETS,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        bucket_count=cfg.bucket_count,
    )
    rng = jax.random.PRNGKey(cfg.seed)
    params, _ = init_policy_params(rng, model)
    return model, params


def _resolve_opponent(cfg: TrainConfig, override: str | None) -> str:
    opponent = (override or cfg.opponent).lower()
    if opponent == "curriculum":
        opponent = "heuristic"
    if opponent not in ("selfplay", "heuristic"):
        raise ValueError(f"unknown opponent mode: {opponent}")
    return opponent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/top1_curriculum.yaml")
    parser.add_argument("--updates", type=int, default=50, help="Number of rollout batches to time.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup batches (not timed).")
    parser.add_argument("--opponent", choices=["selfplay", "heuristic"], default=None)
    parser.add_argument("--no-reset", action="store_true", help="Skip done detection + resets.")
    parser.add_argument("--no-comets", action="store_true", help="Skip host comet spawn checks.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opponent_mode = _resolve_opponent(cfg, args.opponent)

    model, params = _build_model(cfg)
    grid_fn = functools.partial(
        compose_action_grid,
        intercept_iterations=cfg.intercept_iterations,
        enable_planet_block=cfg.enable_planet_block,
    )
    rollout_selfplay = rollout_step_selfplay_factory(model, grid_fn)
    rollout_vs_heuristic = rollout_step_vs_heuristic_factory(model, grid_fn)

    heuristic_agent = None
    if opponent_mode == "heuristic":
        heur_path = None if cfg.heuristic_path in (None, "null") else Path(cfg.heuristic_path)
        heuristic_agent = load_heuristic_agent(heur_path)

    seed_base = cfg.seed * 10000 + 1
    states, learner_players_np = make_initial_states(cfg, seed_base)
    learner_players = jnp.asarray(learner_players_np)
    next_seed = seed_base + cfg.num_envs
    rng = jax.random.PRNGKey(cfg.seed + 1)

    def run_updates(n_updates: int):
        nonlocal states, learner_players, learner_players_np, next_seed, rng
        for _ in range(n_updates):
            for _ in range(cfg.rollout_steps):
                if not args.no_comets:
                    states = maybe_spawn_comets_host(states, cfg)
                rng, sub = jax.random.split(rng)
                if opponent_mode == "selfplay":
                    states, rec, rng = rollout_selfplay(states, params, sub, learner_players)
                else:
                    opp_np = 1 - learner_players_np
                    states, rec, rng = rollout_vs_heuristic(
                        states, params, sub, learner_players, opp_np, heuristic_agent,
                    )
                if args.no_reset:
                    continue
                done_np = np.asarray(rec["done"])
                if done_np.any():
                    states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
                    learner_players_np = np.where(done_np, new_lp, learner_players_np)
                    learner_players = jnp.asarray(learner_players_np)

    # Warmup (compile + cache)
    if args.warmup > 0:
        run_updates(args.warmup)
        jax.block_until_ready(states.step)

    # Timed section
    t0 = time.perf_counter()
    run_updates(args.updates)
    jax.block_until_ready(states.step)
    elapsed = time.perf_counter() - t0

    total_env_steps = cfg.num_envs * cfg.rollout_steps * args.updates
    env_sps = total_env_steps / max(elapsed, 1e-9)

    print(
        "bench_sps | "
        f"envs={cfg.num_envs} rollout={cfg.rollout_steps} updates={args.updates} "
        f"opponent={opponent_mode} intercept_iters={cfg.intercept_iterations} "
        f"planet_block={cfg.enable_planet_block} no_reset={args.no_reset} "
        f"no_comets={args.no_comets} env_sps={env_sps:.1f} elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
