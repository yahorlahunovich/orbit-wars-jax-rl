from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

from kaggle_environments import make


def resolve_agent(root: Path, agent: str) -> str:
    if agent.endswith(".py"):
        return str((root / agent).resolve())
    return agent


def run_game(agent_a: str, agent_b: str, seed: int) -> tuple[float, float, str, str]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    return (
        float(final[0].reward or 0.0),
        float(final[1].reward or 0.0),
        str(final[0].status),
        str(final[1].status),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="versions/kaggle700_current_heuristic/main.py")
    parser.add_argument("--agent-b", default="random")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    agent_a = resolve_agent(root, args.agent_a)
    agent_b = resolve_agent(root, args.agent_b)

    wins = losses = draws = errors = 0
    rewards_a: list[float] = []
    rewards_b: list[float] = []

    t0 = time.perf_counter()
    for seed in range(args.seed_start, args.seed_start + args.games):
        reward_a, reward_b, status_a, status_b = run_game(agent_a, agent_b, seed)
        rewards_a.append(reward_a)
        rewards_b.append(reward_b)

        if status_a != "DONE" or status_b != "DONE":
            errors += 1
        if reward_a > reward_b:
            wins += 1
        elif reward_a < reward_b:
            losses += 1
        else:
            draws += 1
    elapsed = time.perf_counter() - t0

    print(f"A: {args.agent_a}")
    print(f"B: {args.agent_b}")
    print(f"Games: {args.games}")
    print(f"Wins/Losses/Draws: {wins}/{losses}/{draws}")
    print(f"Errors: {errors}")
    print(f"Mean reward A: {mean(rewards_a):.3f}")
    print(f"Mean reward B: {mean(rewards_b):.3f}")
    n = args.games
    per = elapsed / n if n else 0.0
    print(f"Total game time: {elapsed:.3f}s ({per:.3f}s / game)")


if __name__ == "__main__":
    main()
