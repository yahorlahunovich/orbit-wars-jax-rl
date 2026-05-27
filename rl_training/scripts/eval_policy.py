from __future__ import annotations

import argparse
import math
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from direct_runner import run_direct  # noqa: E402
from src.config import TrainConfig, load_train_config  # noqa: E402
from src.eval_utils import build_moves, build_policy, load_checkpoint, resolve_path  # noqa: E402
from src.features import encode_turn  # noqa: E402
from src.heuristic_adapter import heuristic_agent  # noqa: E402

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])


@dataclass(slots=True)
class GameResult:
    seed: int
    policy_side: int
    policy_reward: float
    baseline_reward: float
    elapsed: float

    @property
    def outcome(self) -> str:
        if self.policy_reward > self.baseline_reward:
            return "win"
        if self.policy_reward < self.baseline_reward:
            return "loss"
        return "draw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", choices=("heuristic", "random", "sniper"), required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=4000)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def make_policy_agent(
    cfg: TrainConfig,
    policy: torch.nn.Module,
    device: torch.device,
    deterministic: bool,
) -> Callable[[Any], list[list[float | int]]]:
    def agent(obs: Any) -> list[list[float | int]]:
        batch = encode_turn(obs, cfg.env, env_index=0)
        return build_moves(batch, policy, device, deterministic)

    return agent


def make_baseline_agent(name: str) -> Callable[[Any], list[list[float | int]]]:
    if name == "heuristic":
        return heuristic_agent
    if name == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        def random_wrapped(obs: Any) -> list[list[float | int]]:
            payload = {
                "player": obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0),
                "planets": list(obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])),
            }
            return list(random_agent(payload))

        return random_wrapped
    if name == "sniper":
        return nearest_planet_sniper
    raise ValueError(f"unknown baseline: {name}")


def nearest_planet_sniper(obs: Any) -> list[list[float | int]]:
    moves: list[list[float | int]] = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*planet) for planet in raw_planets]
    my_planets = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    if not targets:
        return moves
    for mine in my_planets:
        nearest = min(targets, key=lambda target: math.hypot(mine.x - target.x, mine.y - target.y))
        ships_needed = max(nearest.ships + 1, 20)
        if mine.ships < ships_needed:
            continue
        angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
        moves.append([mine.id, angle, ships_needed])
    return moves


def run_eval(
    cfg: TrainConfig,
    policy_agent: Callable[[Any], list[list[float | int]]],
    baseline_agent: Callable[[Any], list[list[float | int]]],
    games: int,
    seed_start: int,
    episode_steps: int,
) -> list[GameResult]:
    results: list[GameResult] = []
    for game_idx in range(games):
        seed = seed_start + game_idx
        policy_side = game_idx % 2
        agents = [policy_agent, baseline_agent] if policy_side == 0 else [baseline_agent, policy_agent]
        steps, elapsed = run_direct(agents, seed=seed, episode_steps=episode_steps, keep_steps=False)
        final = steps[-1]
        policy_state = final[policy_side]
        baseline_state = final[1 - policy_side]
        result = GameResult(
            seed=seed,
            policy_side=policy_side,
            policy_reward=float(policy_state.reward or 0.0),
            baseline_reward=float(baseline_state.reward or 0.0),
            elapsed=float(elapsed),
        )
        results.append(result)
        print(
            f"seed={seed} side={policy_side} outcome={result.outcome} "
            f"reward=({result.policy_reward:.3f},{result.baseline_reward:.3f}) "
            f"elapsed={elapsed:.3f}s"
        )
    return results


def summarize(results: list[GameResult]) -> None:
    wins = sum(result.outcome == "win" for result in results)
    draws = sum(result.outcome == "draw" for result in results)
    losses = sum(result.outcome == "loss" for result in results)
    policy_rewards = np.asarray([result.policy_reward for result in results], dtype=np.float64)
    baseline_rewards = np.asarray([result.baseline_reward for result in results], dtype=np.float64)
    elapsed = np.asarray([result.elapsed for result in results], dtype=np.float64)
    print("\n=== Summary ===")
    print(f"games: {len(results)}")
    print(f"wins/draws/losses: {wins}/{draws}/{losses}")
    print(f"win_rate: {wins / max(1, len(results)):.3f}")
    print(f"policy_reward_mean: {policy_rewards.mean():.4f}")
    print(f"baseline_reward_mean: {baseline_rewards.mean():.4f}")
    print(f"reward_diff_mean: {(policy_rewards - baseline_rewards).mean():.4f}")
    print(f"elapsed_total: {elapsed.sum():.3f}s")


def main() -> None:
    args = parse_args()
    cfg = load_train_config(resolve_path(args.config))
    cfg.env.use_fast_env = True
    cfg.env.kaggle_env_root = str(REPO_ROOT / "analysis" / "fast_kaggle_env")
    cfg.env.episode_steps = int(args.episode_steps)

    device = torch.device(args.device)
    policy = build_policy(cfg, device)
    load_checkpoint(policy, resolve_path(args.checkpoint), device)
    policy.eval()
    policy_agent = make_policy_agent(cfg, policy, device, args.deterministic)
    baseline_agent = make_baseline_agent(args.baseline)
    results = run_eval(
        cfg=cfg,
        policy_agent=policy_agent,
        baseline_agent=baseline_agent,
        games=args.games,
        seed_start=args.seed_start,
        episode_steps=args.episode_steps,
    )
    summarize(results)


if __name__ == "__main__":
    main()
