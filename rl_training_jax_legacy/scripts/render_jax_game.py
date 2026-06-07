"""Record one Orbit Wars game with a JAX checkpoint and export HTML replay.

Usage:

    conda run -n ml python rl_training_jax/scripts/render_jax_game.py \
        --checkpoint rl_training_jax/artifacts/jax_ppo_transformer/ckpt_000400.npz \
        --opponent heuristic \
        --seed 100 \
        --output rl_training_jax/artifacts/viz/game.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

# Reuse eval helpers.
sys.path.insert(0, str(ROOT / "scripts"))
from eval_jax_vs_heuristic import _load_heuristic_agent, _load_policy, make_rl_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_selfplay.yaml")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--opponent", choices=("heuristic", "noop", "random"), default="heuristic")
    parser.add_argument("--rl-seat", choices=("0", "1"), default="0", help="Which player seat RL uses")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/viz/game.html")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--kaggle-env-root", type=Path, default=REPO / "analysis/fast_kaggle_env")
    args = parser.parse_args()

    if args.kaggle_env_root.exists():
        sys.path.insert(0, str(args.kaggle_env_root.resolve()))

    import kaggle_environments as ke

    ckpt = args.checkpoint.resolve()
    update = int(np.load(ckpt, allow_pickle=False).get("update", 0))
    params, apply_fn, compose_fn, _ = _load_policy(ckpt, args.config.resolve())
    rl_agent = make_rl_agent(params, apply_fn, compose_fn, deterministic=args.deterministic)

    if args.opponent == "heuristic":
        opp_agent = _load_heuristic_agent()
    elif args.opponent == "noop":
        opp_agent = lambda _obs: []
    else:
        from kaggle_environments.envs.orbit_wars.orbit_wars import agents as ow_agents

        opp_agent = ow_agents["random"]

    agents = [rl_agent, opp_agent] if args.rl_seat == "0" else [opp_agent, rl_agent]

    env = ke.make(
        "orbit_wars",
        configuration={
            "episodeSteps": args.episode_steps,
            "seed": args.seed,
            "actTimeout": 60,
        },
    )
    env.run(agents)

    final = env.state
    r0, r1 = float(final[0].reward), float(final[1].reward)
    rl_r = r0 if args.rl_seat == "0" else r1
    opp_r = r1 if args.rl_seat == "0" else r0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = env.render(mode="html", controls=True)
    args.output.write_text(html, encoding="utf-8")

    json_path = args.json_out or args.output.with_suffix(".json")
    json_path.write_text(json.dumps(env.toJSON(), indent=2), encoding="utf-8")

    print(f"checkpoint update: {update}")
    print(f"seed: {args.seed}  opponent: {args.opponent}  rl_seat: P{args.rl_seat}")
    print(f"final reward: RL={rl_r:+.0f}  opponent={opp_r:+.0f}  steps={len(env.steps)-1}")
    print(f"HTML replay: {args.output.resolve()}")
    print(f"JSON replay: {json_path.resolve()}")
    print("Open the HTML file in your browser to watch the game.")


if __name__ == "__main__":
    main()
