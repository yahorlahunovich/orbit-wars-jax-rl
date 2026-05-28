"""Evaluate a JAX PPO checkpoint vs the kaggle700 heuristic.

Usage (from repo root):

    conda run -n ml python rl_training_jax/scripts/eval_jax_vs_heuristic.py \
        --checkpoint rl_training_jax/artifacts/jax_ppo_transformer/ckpt_000400.npz \
        --config rl_training_jax/configs/transformer_selfplay.yaml \
        --games 20 \
        --deterministic

Also runs RL as player 1 half the time (swap seats).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from orbit_wars import (
    BUCKET_COUNT,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    compose_action_grid,
    encode_observation,
    observation_to_state,
)
from policy import PlanetPolicy
from train_ppo import load_config


def _load_heuristic_agent():
    path = REPO / "versions/kaggle700_current_heuristic/main.py"
    heur_root = path.parent
    sys.path.insert(0, str(heur_root))
    spec = importlib.util.spec_from_file_location("heuristic_main", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.agent


def _load_policy(checkpoint: Path, config: Path):
    cfg = load_config(config)
    model = PlanetPolicy(
        planet_count=MAX_PLANETS,
        fleet_count=MAX_FLEETS,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        bucket_count=cfg.bucket_count,
    )
    from orbit_wars import FLEET_FEATURE_DIM, GLOBAL_FEATURE_DIM, PLANET_FEATURE_DIM

    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
        "fleet_features": jnp.zeros((1, MAX_FLEETS, FLEET_FEATURE_DIM), jnp.float32),
        "fleet_mask": jnp.ones((1, MAX_FLEETS), jnp.bool_),
        "global_features": jnp.zeros((1, GLOBAL_FEATURE_DIM), jnp.float32),
    }
    init_params = model.init(jax.random.PRNGKey(0), **example)
    blob = bytes(np.load(checkpoint, allow_pickle=False)["params"].tobytes())
    params = flax.serialization.from_bytes(init_params, blob)

    @jax.jit
    def compose_fn(state, player):
        return compose_action_grid(state, player)

    @jax.jit
    def apply_fn(p, **kwargs):
        return model.apply(p, **kwargs)

    return params, apply_fn, compose_fn, cfg


def _decode_moves(target_logits, bucket_logits, grid, deterministic: bool) -> list[list[float]]:
    pair_valid = np.asarray(grid["pair_valid"])
    bucket_valid = np.asarray(grid["bucket_valid"])
    source_valid = np.asarray(grid["source_valid"])
    angle_grid = np.asarray(grid["angle"])
    ship_counts = np.asarray(grid["ship_counts"])
    from_ids = np.asarray(grid["from_ids"])
    full_valid = np.asarray(grid["full_valid"])
    target_has_bucket = full_valid.any(axis=-1) & pair_valid

    moves: list[list[float]] = []
    for s in np.where(source_valid)[0]:
        target_mask_s = target_has_bucket[s]
        if not target_mask_s.any():
            continue
        t_logits = np.where(target_mask_s, target_logits[s], -1e9)
        t = int(np.argmax(t_logits))

        bucket_mask_t = full_valid[s, t]
        if not bucket_mask_t.any():
            continue
        b_logits = np.where(bucket_mask_t, bucket_logits[s], -1e9)
        b = int(np.argmax(b_logits)) if deterministic else int(np.argmax(b_logits))

        moves.append([
            float(from_ids[s]),
            float(angle_grid[s, t, b]),
            int(ship_counts[s, t, b]),
        ])
        if len(moves) >= MAX_MOVES_PER_PLAYER:
            break
    return moves


def make_rl_agent(params, apply_fn, compose_fn, deterministic: bool):
    warmed = {"done": False}

    def agent(obs):
        if not warmed["done"]:
            # Warm JIT once.
            player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
            state = observation_to_state(obs)
            feats = encode_observation(state, jnp.int32(player))
            batched = {k: v[None, ...] for k, v in feats.items()}
            _ = apply_fn(params, **batched)
            _ = compose_fn(state, jnp.int32(player))
            warmed["done"] = True

        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        state = observation_to_state(obs)
        feats = encode_observation(state, jnp.int32(player))
        batched = {k: v[None, ...] for k, v in feats.items()}
        out = apply_fn(params, **batched)
        grid = compose_fn(state, jnp.int32(player))
        return _decode_moves(
            np.asarray(out.target_logits[0]),
            np.asarray(out.bucket_logits[0]),
            grid,
            deterministic=deterministic,
        )

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_selfplay.yaml")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--kaggle-env-root", type=Path, default=REPO / "analysis/fast_kaggle_env")
    args = parser.parse_args()

    if args.kaggle_env_root.exists():
        sys.path.insert(0, str(args.kaggle_env_root.resolve()))

    from direct_runner import run_direct

    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    update = int(np.load(ckpt, allow_pickle=False).get("update", 0))
    print(f"Loading checkpoint update={update} from {ckpt}")

    params, apply_fn, compose_fn, cfg = _load_policy(ckpt, args.config.resolve())
    rl_agent = make_rl_agent(params, apply_fn, compose_fn, deterministic=args.deterministic)
    heuristic_agent = _load_heuristic_agent()

    wins = draws = losses = 0
    rl_rewards: list[float] = []

    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.games)):
        rl_is_p0 = (i % 2) == 0
        agents = [rl_agent, heuristic_agent] if rl_is_p0 else [heuristic_agent, rl_agent]
        steps, elapsed = run_direct(
            agents,
            seed=seed,
            episode_steps=args.episode_steps,
            keep_steps=False,
        )
        final = steps[-1]
        r0, r1 = float(final[0].reward), float(final[1].reward)
        rl_r = r0 if rl_is_p0 else r1
        opp_r = r1 if rl_is_p0 else r0
        rl_rewards.append(rl_r)

        if rl_r > opp_r:
            wins += 1
            outcome = "WIN"
        elif rl_r < opp_r:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"

        seat = "P0" if rl_is_p0 else "P1"
        print(
            f"seed={seed} seat={seat} {outcome} "
            f"rl_reward={rl_r:+.0f} opp_reward={opp_r:+.0f} elapsed={elapsed:.2f}s"
        )

    n = max(1, args.games)
    print("\n=== Summary ===")
    print(f"checkpoint update: {update}")
    print(f"games: {args.games}  wins: {wins}  draws: {draws}  losses: {losses}")
    print(f"win_rate: {100.0 * wins / n:.1f}%  draw_rate: {100.0 * draws / n:.1f}%")
    print(f"mean_rl_reward: {np.mean(rl_rewards):+.3f}")
    print(f"deterministic: {args.deterministic}")


if __name__ == "__main__":
    main()
