
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig
from .features import (
    TurnBatch,
    bucket_feature_dim,
    candidate_feature_dim,
    encode_turn,
    global_feature_dim,
    self_feature_dim,
)
from .policy import PlanetPolicy
from .ppo import sample_actions

RL_TRAINING_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (RL_TRAINING_ROOT / candidate).resolve()


def build_policy(cfg: TrainConfig, device: torch.device) -> PlanetPolicy:
    policy = PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        bucket_feature_dim=bucket_feature_dim(),
    ).to(device)
    policy.eval()
    return policy


def load_checkpoint(policy: PlanetPolicy, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()


def build_moves(
    batch: TurnBatch,
    policy: PlanetPolicy,
    device: torch.device,
    deterministic: bool,
) -> list[list[float | int]]:
    if batch.self_features.shape[0] == 0:
        return []
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(batch.self_features).to(device),
            torch.from_numpy(batch.candidate_features).to(device),
            torch.from_numpy(batch.global_features).to(device),
            torch.from_numpy(batch.candidate_mask).to(device).bool(),
            torch.from_numpy(batch.ship_bucket_mask).to(device).bool(),
            torch.from_numpy(batch.bucket_features).to(device),
        )
        sampled = sample_actions(outputs, deterministic=deterministic)
    target_indices = sampled.target_index.detach().cpu().numpy()
    ship_bucket_indices = sampled.ship_bucket_index.detach().cpu().numpy()

    moves: list[list[float | int]] = []
    committed: dict[int, int] = {}
    for row_idx, context in enumerate(batch.contexts):
        target_idx = int(target_indices[row_idx])
        bucket_idx = int(ship_bucket_indices[row_idx])
        if target_idx == 0:
            continue
        if target_idx >= len(context.candidate_ids):
            continue
        if not context.candidate_mask[target_idx]:
            continue
        if bucket_idx < 0 or bucket_idx >= len(context.ship_count_buckets[target_idx]):
            continue
        if not context.ship_bucket_mask[target_idx, bucket_idx]:
            continue
        ships = int(context.ship_count_buckets[target_idx][bucket_idx])
        if ships <= 0:
            continue
        already = committed.get(int(context.source_id), 0)
        source = next((planet for planet in batch.state.planets if planet.id == context.source_id), None)
        if source is None or int(source.ships) < already + ships:
            continue
        moves.append([context.source_id, float(context.target_angles[target_idx]), ships])
        committed[int(context.source_id)] = already + ships
    return moves


def make_policy_agent(
    cfg: TrainConfig,
    policy: PlanetPolicy,
    device: torch.device,
    deterministic: bool,
):
    def agent(obs: Any) -> list[list[float | int]]:
        batch = encode_turn(obs, cfg.env, env_index=0)
        return build_moves(batch, policy, device, deterministic)

    return agent
