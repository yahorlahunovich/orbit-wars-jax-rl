#!/usr/bin/env python3
"""Benchmark direct Orbit Wars runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from direct_runner import run_direct_from_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", default="noop")
    parser.add_argument("--agent-b", default="noop")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--kaggle-env-root", type=Path, default=None)
    parser.add_argument("--keep-steps", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.kaggle_env_root is not None:
        sys.path.insert(0, str(args.kaggle_env_root.resolve()))

    total_elapsed = 0.0
    total_steps = 0
    for seed in range(args.seed_start, args.seed_start + args.games):
        steps, elapsed = run_direct_from_names(
            [args.agent_a, args.agent_b],
            root=root,
            seed=seed,
            episode_steps=args.episode_steps,
            keep_steps=args.keep_steps,
        )
        step_count = len(steps) if args.keep_steps else args.episode_steps
        total_steps += step_count
        total_elapsed += elapsed
        final = steps[-1]
        print(
            f"seed={seed} steps={step_count} elapsed={elapsed:.6f}s "
            f"reward=({final[0].reward},{final[1].reward}) "
            f"status=({final[0].status},{final[1].status})"
        )

    print("\n=== Summary ===")
    print(f"games: {args.games}")
    print(f"steps: {total_steps}")
    print(f"elapsed_total: {total_elapsed:.6f}s")
    print(f"elapsed_per_game: {total_elapsed / max(1, args.games):.6f}s")
    print(f"elapsed_per_step: {1000.0 * total_elapsed / max(1, total_steps):.3f}ms")


if __name__ == "__main__":
    main()
