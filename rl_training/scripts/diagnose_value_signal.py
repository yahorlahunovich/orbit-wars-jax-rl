#!/usr/bin/env python3
"""Measure how well global features predict terminal return (proxy for critic signal)."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from src.config import load_train_config
from src.env import OrbitWarsEnv
from src.features import build_global_features, candidate_feature_dim, encode_turn, global_feature_dim, self_feature_dim
from src.opponents import build_opponent
from src.ppo import _explained_variance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_ppo.yaml")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=9000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(RL_ROOT / args.config)
    opponent = build_opponent(cfg.opponent, cfg)
    env = OrbitWarsEnv(cfg, opponent, env_index=0)

    global_rows: list[np.ndarray] = []
    returns: list[float] = []
    per_turn_values: list[list[float]] = []

    for game_idx in range(args.games):
        batch = env.reset(seed=args.seed_start + game_idx)
        episode_return = 0.0
        turn_globals: list[np.ndarray] = []
        steps = 0
        while True:
            if batch.self_features.shape[0] > 0:
                turn_globals.append(build_global_features(batch.state, cfg.env))
            moves = []
            result = env.step(moves)
            episode_return += float(result.reward)
            steps += 1
            if result.done:
                break
            batch = result.batch
        for global_feat in turn_globals:
            global_rows.append(global_feat)
            returns.append(float(episode_return))
        per_turn_values.append([float(episode_return)] * len(turn_globals))

    global_matrix = np.asarray(global_rows, dtype=np.float32)
    return_array = np.asarray(returns, dtype=np.float32)

    # Linear baseline: military lead + production lead should correlate with outcome.
    military_lead = global_matrix[:, 10]
    production_lead = global_matrix[:, 11]
    linear_pred = 0.6 * military_lead + 0.4 * production_lead
    linear_pred = linear_pred * float(return_array.std() + 1e-8) + float(return_array.mean())

    import torch

    ev_leads = _explained_variance(
        torch.tensor(return_array),
        torch.tensor(linear_pred, dtype=torch.float32),
    )
    ev_zero = _explained_variance(
        torch.tensor(return_array),
        torch.zeros_like(torch.tensor(return_array)),
    )

    same_turn_groups = [len(group) for group in per_turn_values if group]
    print(f"games={args.games} decision_rows={len(returns)} avg_rows_per_turn={np.mean(same_turn_groups):.2f}")
    print(f"return_std={return_array.std():.4f} return_mean={return_array.mean():.4f}")
    print(f"explained_variance(linear_lead_baseline)={ev_leads:.4f}")
    print(f"explained_variance(zero_baseline)={ev_zero:.4f}")
    print(
        "full policy uses "
        f"self={self_feature_dim()} candidate={candidate_feature_dim()} global={global_feature_dim()}"
    )


if __name__ == "__main__":
    main()
