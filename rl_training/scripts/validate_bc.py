from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from direct_runner import run_direct  # noqa: E402
from eval_policy import make_baseline_agent  # noqa: E402
from src.config import TrainConfig  # noqa: E402
from src.eval_utils import build_moves, build_policy, load_checkpoint, resolve_path  # noqa: E402
from src.features import encode_turn  # noqa: E402


def run_bc_validation(
    *,
    cfg: TrainConfig,
    policy: torch.nn.Module,
    device: torch.device,
    games: int = 20,
    seed_start: int = 5000,
    episode_steps: int = 200,
    min_wins: int = 15,
) -> dict[str, Any]:
    cfg.env.use_fast_env = True
    cfg.env.kaggle_env_root = str(REPO_ROOT / "analysis" / "fast_kaggle_env")
    cfg.env.episode_steps = int(episode_steps)

    policy.eval()
    baseline_agent = make_baseline_agent("random")

    def policy_agent(obs: Any) -> list[list[float | int]]:
        batch = encode_turn(obs, cfg.env, env_index=0)
        return build_moves(batch, policy, device, deterministic=True)

    wins = 0
    total_sends = 0
    friendly_sends = 0
    turns_with_send = 0

    for game_idx in range(games):
        seed = seed_start + game_idx
        policy_side = game_idx % 2
        agents = [policy_agent, baseline_agent] if policy_side == 0 else [baseline_agent, policy_agent]
        steps, _elapsed = run_direct(agents, seed=seed, episode_steps=episode_steps, keep_steps=True)

        final = steps[-1]
        policy_state = final[policy_side]
        baseline_state = final[1 - policy_side]
        if float(policy_state.reward or 0.0) > float(baseline_state.reward or 0.0):
            wins += 1

        for step in steps:
            agent_state = step[policy_side]
            obs = agent_state.observation
            if obs is None:
                continue
            batch = encode_turn(obs, cfg.env, env_index=0)
            if batch.self_features.shape[0] == 0:
                continue
            moves = build_moves(batch, policy, device, deterministic=True)
            if not moves:
                continue
            turns_with_send += 1
            planet_by_id = {planet.id: planet for planet in batch.state.planets}
            for move in moves:
                total_sends += 1
                src_id = int(move[0])
                angle = float(move[1])
                src = planet_by_id.get(src_id)
                if src is None:
                    continue
                best_tgt = None
                best_delta = float("inf")
                for planet in batch.state.planets:
                    if planet.id == src_id:
                        continue
                    bearing = math.atan2(planet.y - src.y, planet.x - src.x)
                    delta = abs(((angle - bearing + math.pi) % (2 * math.pi)) - math.pi)
                    if delta < best_delta:
                        best_delta = delta
                        best_tgt = planet
                if best_tgt is not None and best_tgt.owner == batch.state.player:
                    friendly_sends += 1

    send_rate = total_sends / max(games * episode_steps, 1)
    passed = wins >= min_wins and total_sends > 0

    return {
        "games": games,
        "wins_vs_random": wins,
        "min_wins_required": min_wins,
        "send_rate": send_rate,
        "send_turn_rate": turns_with_send / max(games * episode_steps, 1),
        "total_sends": total_sends,
        "friendly_target_sends": friendly_sends,
        "friendly_target_send_rate": friendly_sends / max(total_sends, 1),
        "passed": passed,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="artifacts/bc/bc_best.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-wins", type=int, default=15)
    args = parser.parse_args()

    from src.config import load_train_config

    cfg = load_train_config(resolve_path(args.config))
    device = torch.device(args.device)
    policy = build_policy(cfg, device)
    load_checkpoint(policy, resolve_path(args.checkpoint), device)
    result = run_bc_validation(
        cfg=cfg,
        policy=policy,
        device=device,
        games=args.games,
        seed_start=args.seed_start,
        episode_steps=args.episode_steps,
        min_wins=args.min_wins,
    )
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
