from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from src.config import load_train_config  # noqa: E402
from src.features import bucket_feature_dim, candidate_feature_dim, global_feature_dim, self_feature_dim  # noqa: E402
from src.policy import PlanetPolicy  # noqa: E402
from validate_bc import run_bc_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="artifacts/bc/top_players_bc.npz")
    parser.add_argument("--output", default="artifacts/bc/bc_best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-balance-no-ops", action="store_true")
    parser.add_argument("--validate-games", type=int, default=20)
    parser.add_argument("--validate-seed-start", type=int, default=5000)
    parser.add_argument("--validate-episode-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_train_config(resolve_path(args.config))
    device = torch.device(args.device)
    data = np.load(resolve_path(args.dataset))

    self_features = torch.from_numpy(data["self_features"]).float()
    candidate_features = torch.from_numpy(data["candidate_features"]).float()
    global_features = torch.from_numpy(data["global_features"]).float()
    candidate_mask = torch.from_numpy(data["candidate_mask"]).bool()
    target_index = torch.from_numpy(data["target_index"]).long()
    has_bucket_labels = "ship_bucket_index" in data
    ship_bucket_index = (
        torch.from_numpy(data["ship_bucket_index"]).long()
        if has_bucket_labels
        else torch.zeros_like(target_index)
    )
    has_bucket_features = "ship_bucket_mask" in data and "bucket_features" in data
    ship_bucket_mask = (
        torch.from_numpy(data["ship_bucket_mask"]).bool()
        if has_bucket_features
        else torch.ones(
            target_index.shape[0],
            cfg.env.candidate_count,
            cfg.env.ship_bucket_count,
            dtype=torch.bool,
        )
    )
    bucket_features = (
        torch.from_numpy(data["bucket_features"]).float()
        if has_bucket_features
        else torch.zeros(
            target_index.shape[0],
            cfg.env.candidate_count,
            cfg.env.ship_bucket_count,
            bucket_feature_dim(),
        )
    )

    policy = PlanetPolicy(
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        bucket_feature_dim=bucket_feature_dim(),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))

    size = int(target_index.shape[0])
    order = torch.randperm(size)
    val_size = int(size * float(args.val_fraction))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    if not args.no_balance_no_ops:
        train_idx = balance_train_indices(target_index, train_idx, args.seed)
    action_rows = int((target_index > 0).sum())
    no_op_rows = int((target_index == 0).sum())
    print(
        f"rows={size} train={len(train_idx)} val={len(val_idx)} "
        f"action_rows={action_rows} no_op_rows={no_op_rows} "
        f"bucket_labels={has_bucket_labels} bucket_features={has_bucket_features}"
    )

    best_val_loss = float("inf")
    best_path = resolve_path(args.output)

    for epoch in range(1, int(args.epochs) + 1):
        policy.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        train_target_loss = 0.0
        train_bucket_loss = 0.0
        train_target_correct = 0
        train_bucket_correct = 0
        train_total = 0
        for start in range(0, len(shuffled), int(args.batch_size)):
            idx = shuffled[start : start + int(args.batch_size)]
            outputs = policy(
                self_features[idx].to(device),
                candidate_features[idx].to(device),
                global_features[idx].to(device),
                candidate_mask[idx].to(device),
                ship_bucket_mask[idx].to(device),
                bucket_features[idx].to(device),
            )
            tgt_labels = target_index[idx].to(device)
            bkt_labels = ship_bucket_index[idx].to(device)

            target_loss = F.cross_entropy(outputs.target_logits, tgt_labels)
            row_idx = torch.arange(outputs.ship_bucket_logits.shape[0], device=device)
            selected_bucket_logits = outputs.ship_bucket_logits[row_idx, tgt_labels]
            bucket_loss = F.cross_entropy(selected_bucket_logits, bkt_labels)
            loss = target_loss + bucket_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            batch_size = int(idx.shape[0])
            train_target_loss += float(target_loss.detach().cpu()) * batch_size
            train_bucket_loss += float(bucket_loss.detach().cpu()) * batch_size
            train_target_correct += int((outputs.target_logits.argmax(dim=-1) == tgt_labels).sum().detach().cpu())
            train_bucket_correct += int(
                (selected_bucket_logits.argmax(dim=-1) == bkt_labels).sum().detach().cpu()
            )
            train_total += batch_size

        val_metrics = evaluate(
            policy,
            self_features,
            candidate_features,
            global_features,
            candidate_mask,
            ship_bucket_mask,
            bucket_features,
            target_index,
            ship_bucket_index,
            val_idx,
            int(args.batch_size),
            device,
        )
        val_loss = val_metrics["target_loss"] + val_metrics["bucket_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(policy, cfg, args, size, has_bucket_labels, best_path)

        print(
            f"epoch={epoch} "
            f"target_loss={train_target_loss / max(train_total, 1):.4f} "
            f"bucket_loss={train_bucket_loss / max(train_total, 1):.4f} "
            f"target_acc={train_target_correct / max(train_total, 1):.4f} "
            f"bucket_acc={train_bucket_correct / max(train_total, 1):.4f} "
            f"val_target_loss={val_metrics['target_loss']:.4f} "
            f"val_bucket_loss={val_metrics['bucket_loss']:.4f} "
            f"val_target_acc={val_metrics['target_acc']:.4f} "
            f"val_bucket_acc={val_metrics['bucket_acc']:.4f} "
            f"val_total_loss={val_loss:.4f}"
        )

    if not best_path.exists():
        save_checkpoint(policy, cfg, args, size, has_bucket_labels, best_path)
    print(f"wrote={best_path}")

    if args.skip_validation:
        return

    validation = run_bc_validation(
        cfg=cfg,
        policy=policy,
        device=device,
        games=int(args.validate_games),
        seed_start=int(args.validate_seed_start),
        episode_steps=int(args.validate_episode_steps),
    )
    validation_path = best_path.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise SystemExit(
            "BC validation failed: need >=15/20 wins vs random and total_sends > 0 "
            "before starting PPO."
        )


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (RL_ROOT / candidate).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balance_train_indices(
    target_index: torch.Tensor,
    train_idx: torch.Tensor,
    seed: int,
    *,
    no_op_multiplier: float = 2.0,
) -> torch.Tensor:
    """Cap no-op rows so action supervision is not drowned out during BC."""
    train_targets = target_index[train_idx]
    action_mask = train_targets > 0
    action_idx = train_idx[action_mask]
    no_op_idx = train_idx[~action_mask]
    if len(action_idx) == 0 or len(no_op_idx) == 0:
        return train_idx
    cap = max(int(len(action_idx) * no_op_multiplier), len(action_idx))
    if len(no_op_idx) > cap:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        pick = torch.randperm(len(no_op_idx), generator=generator)[:cap]
        no_op_idx = no_op_idx[pick]
    balanced = torch.cat([action_idx, no_op_idx])
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1)
    return balanced[torch.randperm(len(balanced), generator=generator)]


def save_checkpoint(
    policy: PlanetPolicy,
    cfg: Any,
    args: argparse.Namespace,
    size: int,
    has_bucket_labels: bool,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "config": cfg,
            "bc": {
                "dataset": str(resolve_path(args.dataset)),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "has_bucket_labels": has_bucket_labels,
            },
        },
        output,
    )


def evaluate(
    policy: PlanetPolicy,
    self_features: torch.Tensor,
    candidate_features: torch.Tensor,
    global_features: torch.Tensor,
    candidate_mask: torch.Tensor,
    ship_bucket_mask: torch.Tensor,
    bucket_features: torch.Tensor,
    target_index: torch.Tensor,
    ship_bucket_index: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if len(indices) == 0:
        return {
            "target_loss": 0.0,
            "bucket_loss": 0.0,
            "target_acc": 0.0,
            "bucket_acc": 0.0,
        }
    policy.eval()
    total_target_loss = 0.0
    total_bucket_loss = 0.0
    target_correct = 0
    bucket_correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            outputs = policy(
                self_features[idx].to(device),
                candidate_features[idx].to(device),
                global_features[idx].to(device),
                candidate_mask[idx].to(device),
                ship_bucket_mask[idx].to(device),
                bucket_features[idx].to(device),
            )
            labels = target_index[idx].to(device)
            bkt_labels = ship_bucket_index[idx].to(device)
            target_loss = F.cross_entropy(outputs.target_logits, labels)
            row_idx = torch.arange(outputs.ship_bucket_logits.shape[0], device=device)
            selected = outputs.ship_bucket_logits[row_idx, labels]
            bucket_loss = F.cross_entropy(selected, bkt_labels)
            count = int(idx.shape[0])
            total_target_loss += float(target_loss.detach().cpu()) * count
            total_bucket_loss += float(bucket_loss.detach().cpu()) * count
            target_correct += int((outputs.target_logits.argmax(dim=-1) == labels).sum().detach().cpu())
            bucket_correct += int((selected.argmax(dim=-1) == bkt_labels).sum().detach().cpu())
            total += count
    return {
        "target_loss": total_target_loss / max(total, 1),
        "bucket_loss": total_bucket_loss / max(total, 1),
        "target_acc": target_correct / max(total, 1),
        "bucket_acc": bucket_correct / max(total, 1),
    }


if __name__ == "__main__":
    main()
