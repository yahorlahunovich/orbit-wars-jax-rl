#!/usr/bin/env python3
"""Report decode-level angle stats for a checkpoint (ships, buckets, masks).

Usage:
    PYTHONPATH=src:scripts python scripts/analyze_angle_outcomes.py \\
        --checkpoint artifacts/jax_ppo_transformer/ckpt_000400.npz \\
        --seeds 100 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_jax_vs_heuristic import _load_policy
from orbit_wars import encode_observation, observation_to_state, reset, step


def analyze_seed(params, apply_fn, compose_fn, seed: int, steps: int, player: int = 0) -> dict:
    state = reset(seed, episode_steps=500)
    ship_hist = Counter()
    bucket_hist = Counter()
    mask_hist = Counter()
    n_moves = 0

    for _ in range(steps):
        feats = encode_observation(state, jnp.int32(player))
        out = apply_fn(params, **{k: v[None, ...] for k, v in feats.items()})
        grid = compose_fn(state, jnp.int32(player))
        tl = np.asarray(out.target_logits[0])
        bl = np.asarray(out.bucket_logits[0])
        bv = np.asarray(grid["bucket_valid"])
        pv = np.asarray(grid["pair_valid"])
        sv = np.asarray(grid["source_valid"])
        thb = bv.any(-1) & pv

        rows = []
        for s in np.where(sv)[0]:
            tm = thb[s]
            if not tm.any():
                continue
            t = int(np.argmax(np.where(tm, tl[s], -1e9)))
            bm = bv[s, t]
            if not bm.any():
                continue
            b = int(np.argmax(np.where(bm, bl[s], -1e9)))
            ships = int(grid["ship_counts"][s, t, b])
            ship_hist[ships] += 1
            bucket_hist[b] += 1
            valid = bool(grid["full_valid"][s, t, b])
            if not valid:
                if grid["sun_blocks"][s, t, b]:
                    mask_hist["sun"] += 1
                elif grid["planet_blocks"][s, t, b]:
                    mask_hist["planet"] += 1
                else:
                    mask_hist["bucket/other"] += 1
            if valid:
                rows.append([float(grid["from_ids"][s]), float(grid["angle"][s, t, b]), ships])
                n_moves += 1

        state = step(state, [rows, []])

    valid_total = sum(ship_hist.values()) - sum(mask_hist.values())
    return {
        "seed": seed,
        "valid_moves": n_moves,
        "ships": dict(sorted(ship_hist.items())),
        "buckets": dict(sorted(bucket_hist.items())),
        "masked": dict(mask_hist),
        "pct_1ship": ship_hist[1] / max(1, sum(ship_hist.values())) * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/transformer_selfplay.yaml")
    p.add_argument("--seeds", type=str, default="100")
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    params, apply_fn, compose_fn, _ = _load_policy(args.checkpoint, args.config)
    print(f"Decode stats — {args.checkpoint.name}")
    for seed in [int(s) for s in args.seeds.split(",")]:
        s = analyze_seed(params, apply_fn, compose_fn, seed, args.steps)
        print(f"\nseed={s['seed']}  valid_moves={s['valid_moves']}")
        print(f"  ships={s['ships']}")
        print(f"  buckets={s['buckets']}")
        print(f"  masked={s['masked']}")
        print(f"  1-ship share={s['pct_1ship']:.1f}%")


if __name__ == "__main__":
    main()
