"""JAX inference for the Kaggle Orbit Wars submission.

Loads weights once on first call (lazily — Kaggle re-imports per game), then
runs `obs -> state -> features -> policy -> action list` per step.

The decoded action list is `[[from_planet_id, angle_radians, num_ships], ...]`,
matching what the Kaggle harness expects.

Inference is deterministic (argmax over target and bucket), which is usually
slightly stronger than sampling in competitive matches.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.orbit_wars import (
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    compose_action_grid,
    encode_observation,
    observation_to_state,
)
from src.policy import PlanetPolicy

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "policy.msgpack"
_MODEL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "weights" / "model_config.json"

_state = {
    "model": None,
    "params": None,
    "apply_fn": None,
    "compose_fn": None,
}


def _load() -> None:
    """One-time load of model + params. Called on first `agent()` invocation."""
    import json

    import flax.serialization

    cfg = json.loads(_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model = PlanetPolicy(
        planet_count=MAX_PLANETS,
        fleet_count=MAX_FLEETS,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        bucket_count=cfg["bucket_count"],
    )
    raw = _WEIGHTS_PATH.read_bytes()

    # Initialize a dummy params tree to give flax.serialization a target.
    init_rng = jax.random.PRNGKey(0)
    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, cfg["planet_feature_dim"]), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
    }
    init_params = model.init(init_rng, **example)
    params = flax.serialization.from_bytes(init_params, raw)

    @jax.jit
    def apply_fn(p, **kwargs):
        return model.apply(p, **kwargs)

    @jax.jit
    def compose_fn(state, player):
        return compose_action_grid(state, player)

    # Warm up jit by running one pass.
    _ = apply_fn(params, **example)

    _state["model"] = model
    _state["params"] = params
    _state["apply_fn"] = apply_fn
    _state["compose_fn"] = compose_fn


def _decode_moves(
    target_logits: np.ndarray,    # (P, P)
    bucket_logits: np.ndarray,    # (P, BUCKETS)
    grid: dict,
) -> list[list[float]]:
    """Argmax-decode actions for one env. Apply all masks; emit at most
    `MAX_MOVES_PER_PLAYER` legal moves as Kaggle expects."""
    pair_valid = np.asarray(grid["pair_valid"])               # (P, P)
    bucket_valid = np.asarray(grid["bucket_valid"])           # (P, P, B)
    source_valid = np.asarray(grid["source_valid"])           # (P,)
    angle_grid = np.asarray(grid["angle"])                    # (P, P, B)
    ship_counts = np.asarray(grid["ship_counts"])             # (P, P, B)
    from_ids = np.asarray(grid["from_ids"])                   # (P,)

    target_has_bucket = bucket_valid.any(axis=-1) & pair_valid    # (P, P)
    moves: list[list[float]] = []

    for s in np.where(source_valid)[0]:
        target_mask_s = target_has_bucket[s]                   # (P,)
        if not target_mask_s.any():
            continue
        # Mask invalid targets with -inf before argmax.
        t_logits = np.where(target_mask_s, target_logits[s], -1e9)
        t = int(np.argmax(t_logits))

        bucket_mask_t = bucket_valid[s, t] & np.asarray(grid["full_valid"])[s, t]                     # (B,)
        if not bucket_mask_t.any():
            continue
        b_logits = np.where(bucket_mask_t, bucket_logits[s], -1e9)
        b = int(np.argmax(b_logits))

        moves.append([
            float(from_ids[s]),
            float(angle_grid[s, t, b]),
            int(ship_counts[s, t, b]),
        ])
        if len(moves) >= MAX_MOVES_PER_PLAYER:
            break

    return moves


def agent(obs: Any) -> list[list[int | float]]:
    """Kaggle agent entry. `obs` is a dict-or-attr-style observation."""
    if _state["model"] is None:
        _load()

    try:
        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        state = observation_to_state(obs)

        # Add batch dim of 1.
        feats = encode_observation(state, jnp.int32(player))
        batched_feats = {k: v[None, ...] for k, v in feats.items()}
        out = _state["apply_fn"](_state["params"], **batched_feats)
        grid = _state["compose_fn"](state, jnp.int32(player))

        target_logits = np.asarray(out.target_logits[0])
        bucket_logits = np.asarray(out.bucket_logits[0])

        return _decode_moves(target_logits, bucket_logits, grid)
    except Exception as exc:
        # Defensive fallback: never crash the Kaggle harness.
        if os.environ.get("ORBIT_WARS_DEBUG"):
            import traceback
            traceback.print_exc()
        return []
