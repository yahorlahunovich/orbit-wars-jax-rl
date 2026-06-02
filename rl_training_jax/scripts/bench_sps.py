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
from orbit_wars.heuristic_opponent import batched_heuristic_actions
from orbit_wars.step import step_jit
from orbit_wars.heuristic_opponent import load_heuristic_agent
from policy import PlanetPolicy
from train_ppo import (
    TrainConfig,
    init_policy_params,
    learner_record_from_single,
    make_initial_states,
    maybe_spawn_comets_host,
    reset_done_envs,
    rollout_step_selfplay_factory,
    rollout_step_vs_heuristic_factory,
    sample_learner_factory,
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
    parser.add_argument("--num-envs", type=int, default=None, help="Override env.num_envs for quick tests.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override env.rollout_steps.")
    parser.add_argument("--episode-steps", type=int, default=None, help="Override env.episode_steps.")
    parser.add_argument(
        "--intercept-iterations",
        type=int,
        default=None,
        help="Override action-grid intercept iterations.",
    )
    incoming_proj = parser.add_mutually_exclusive_group()
    incoming_proj.add_argument(
        "--enable-incoming-projection",
        action="store_true",
        help="Enable incoming fleet projections in action grid.",
    )
    incoming_proj.add_argument(
        "--disable-incoming-projection",
        action="store_true",
        help="Disable incoming fleet projections (faster).",
    )
    planet_block = parser.add_mutually_exclusive_group()
    planet_block.add_argument("--enable-planet-block", action="store_true", help="Enable planet block checks.")
    planet_block.add_argument("--disable-planet-block", action="store_true", help="Disable planet block checks.")
    parser.add_argument("--no-reset", action="store_true", help="Skip done detection + resets.")
    parser.add_argument("--no-comets", action="store_true", help="Skip host comet spawn checks.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print rough timing breakdown (heuristic mode only).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_envs is not None:
        cfg.num_envs = int(args.num_envs)
    if args.rollout_steps is not None:
        cfg.rollout_steps = int(args.rollout_steps)
    if args.episode_steps is not None:
        cfg.episode_steps = int(args.episode_steps)
    if args.intercept_iterations is not None:
        cfg.intercept_iterations = int(args.intercept_iterations)
    if args.enable_planet_block:
        cfg.enable_planet_block = True
    if args.disable_planet_block:
        cfg.enable_planet_block = False
    if args.enable_incoming_projection:
        cfg.enable_incoming_projection = True
    if args.disable_incoming_projection:
        cfg.enable_incoming_projection = False
    opponent_mode = _resolve_opponent(cfg, args.opponent)

    model, params = _build_model(cfg)
    grid_fn = functools.partial(
        compose_action_grid,
        intercept_iterations=cfg.intercept_iterations,
        enable_planet_block=cfg.enable_planet_block,
        enable_incoming_projection=cfg.enable_incoming_projection,
    )
    rollout_selfplay = rollout_step_selfplay_factory(model, grid_fn)
    rollout_vs_heuristic = rollout_step_vs_heuristic_factory(model, grid_fn)
    sample_learner = sample_learner_factory(model, grid_fn)

    heuristic_agent = None
    if opponent_mode == "heuristic":
        heur_path = None if cfg.heuristic_path in (None, "null") else Path(cfg.heuristic_path)
        heuristic_agent = load_heuristic_agent(heur_path)

    seed_base = cfg.seed * 10000 + 1
    
    # Generate reset pool
    import dataclasses
    cfg_pool = dataclasses.replace(cfg, num_envs=256)
    reset_pool, _ = make_initial_states(cfg_pool, cfg.seed + 100000)

    states, learner_players_np = make_initial_states(cfg, seed_base)
    learner_players = jnp.asarray(learner_players_np)
    next_seed = seed_base + cfg.num_envs

    rng = jax.random.PRNGKey(cfg.seed + 1)

    def run_updates(n_updates: int):
        nonlocal states, learner_players, learner_players_np, next_seed, rng
        for _ in range(n_updates):
            for _ in range(cfg.rollout_steps):
                if not args.no_comets and opponent_mode != "selfplay":
                    states = maybe_spawn_comets_host(states, cfg)
                rng, sub = jax.random.split(rng)
                if opponent_mode == "selfplay":
                    states, rec, rng, learner_players = rollout_selfplay(states, params, sub, learner_players, reset_pool)
                else:
                    opp_np = 1 - learner_players_np
                    states, rec, rng = rollout_vs_heuristic(
                        states, params, sub, learner_players, opp_np, heuristic_agent,
                    )
                if args.no_reset or opponent_mode == "selfplay":
                    continue
                done_np = np.asarray(rec["done"])
                if done_np.any():
                    states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
                    learner_players_np = np.where(done_np, new_lp, learner_players_np)
                    learner_players = jnp.asarray(learner_players_np)

    def run_updates_profile(n_updates: int):
        nonlocal states, learner_players, learner_players_np, next_seed, rng
        timers = {
            "comets": 0.0,
            "policy_grid": 0.0,
            "heuristic": 0.0,
            "step": 0.0,
            "reset": 0.0,
        }
        for _ in range(n_updates):
            for _ in range(cfg.rollout_steps):
                if not args.no_comets and opponent_mode != "selfplay":
                    t0 = time.perf_counter()
                    states = maybe_spawn_comets_host(states, cfg)
                    timers["comets"] += time.perf_counter() - t0

                rng, sub = jax.random.split(rng)

                t0 = time.perf_counter()
                actions, mask, sampled, out, grid, feats, rng = sample_learner(
                    states, params, sub, learner_players,
                )
                jax.block_until_ready(out.value)
                timers["policy_grid"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                if opponent_mode == "heuristic":
                    opp_np = 1 - learner_players_np
                    ha0, hm0, ha1, hm1 = batched_heuristic_actions(states, opp_np, heuristic_agent)
                else:
                    ha0, hm0, ha1, hm1 = actions, mask, actions, mask  # Profile doesn't matter for this mode
                timers["heuristic"] += time.perf_counter() - t0

                is_learner_p0 = (learner_players == 0)
                final_a0 = jnp.where(is_learner_p0[:, None, None], actions, ha0)
                final_a1 = jnp.where(is_learner_p0[:, None, None], ha1, actions)
                final_m0 = jnp.where(is_learner_p0[:, None], mask, hm0)
                final_m1 = jnp.where(is_learner_p0[:, None], hm1, mask)

                t0 = time.perf_counter()
                new_states = jax.vmap(step_jit)(states, final_a0, final_a1, final_m0, final_m1)
                record = learner_record_from_single(
                    learner_players, sampled, out, grid, feats, new_states,
                )
                jax.block_until_ready(record["done"])
                timers["step"] += time.perf_counter() - t0

                states = new_states

                if args.no_reset or opponent_mode == "selfplay":
                    continue
                done_np = np.asarray(record["done"])
                if done_np.any():
                    t0 = time.perf_counter()
                    states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
                    learner_players_np = np.where(done_np, new_lp, learner_players_np)
                    learner_players = jnp.asarray(learner_players_np)
                    timers["reset"] += time.perf_counter() - t0
        return timers

    # Warmup (compile + cache)
    if args.warmup > 0:
        if args.profile and opponent_mode == "heuristic":
            run_updates_profile(args.warmup)
        else:
            run_updates(args.warmup)
        jax.block_until_ready(states.step)

    # Timed section
    t0 = time.perf_counter()
    timers = None
    if args.profile and opponent_mode == "heuristic":
        timers = run_updates_profile(args.updates)
        jax.block_until_ready(states.step)
    else:
        run_updates(args.updates)
        jax.block_until_ready(states.step)
    elapsed = time.perf_counter() - t0

    total_env_steps = cfg.num_envs * cfg.rollout_steps * args.updates
    env_sps = total_env_steps / max(elapsed, 1e-9)

    print(
        "bench_sps | "
        f"envs={cfg.num_envs} rollout={cfg.rollout_steps} updates={args.updates} "
        f"opponent={opponent_mode} intercept_iters={cfg.intercept_iterations} "
        f"planet_block={cfg.enable_planet_block} incoming_proj={cfg.enable_incoming_projection} "
        f"no_reset={args.no_reset} "
        f"no_comets={args.no_comets} env_sps={env_sps:.1f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    if timers:
        total = sum(timers.values()) or 1e-9
        print(
            "bench_sps_breakdown | "
            f"comets={timers['comets']:.3f}s "
            f"policy_grid={timers['policy_grid']:.3f}s "
            f"heuristic={timers['heuristic']:.3f}s "
            f"step={timers['step']:.3f}s "
            f"reset={timers['reset']:.3f}s "
            f"(pct: "
            f"comets={timers['comets']/total:.0%}, "
            f"policy_grid={timers['policy_grid']/total:.0%}, "
            f"heuristic={timers['heuristic']/total:.0%}, "
            f"step={timers['step']/total:.0%}, "
            f"reset={timers['reset']/total:.0%})",
            flush=True,
        )


if __name__ == "__main__":
    main()
