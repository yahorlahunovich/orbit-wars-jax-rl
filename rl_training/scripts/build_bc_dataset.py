from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from src.config import load_train_config  # noqa: E402
from src.features import (  # noqa: E402
    bucket_feature_dim,
    bucket_index_for_ships,
    build_bucket_features,
    build_candidate_features,
    build_feature_cache,
    build_global_features,
    build_self_features,
    candidate_feature_dim,
    candidate_index_for_target,
    planet_slot_for_id,
    resolve_action_target_id,
    self_feature_dim,
    target_slot_count,
)
from src.game_types import GameState, parse_observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir",
        default="/media/yahor/ADATA SE880/datasets/orbit-wars/replays_raw/top_players",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts/bc/top_players_bc.npz")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--angle-threshold", type=float, default=0.35)
    parser.add_argument("--no-op-ratio", type=float, default=1.0)
    parser.add_argument("--include-losers", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    cfg = load_train_config(resolve_path(args.config))
    replay_dir = Path(args.replay_dir)
    files = sorted(replay_dir.glob("*.json"))
    if args.max_files > 0:
        files = files[: args.max_files]

    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    ship_bucket_mask_rows: list[np.ndarray] = []
    bucket_feat_rows: list[np.ndarray] = []
    labels: list[int] = []
    bucket_labels: list[int] = []
    stats: Counter[str] = Counter()

    for path in files:
        episode = json.loads(path.read_text(encoding="utf-8"))
        rewards = episode.get("rewards", [])
        winners = {idx for idx, reward in enumerate(rewards) if reward == 1}
        steps = episode.get("steps", [])
        if len(steps) < 2:
            continue

        for step_idx in range(1, len(steps)):
            fallback_step = (steps[step_idx - 1][0].get("observation") or {}).get("step", step_idx - 1)
            for agent_idx, agent_state in enumerate(steps[step_idx]):
                if not args.include_losers and agent_idx not in winners:
                    continue
                actions = valid_actions(agent_state.get("action") or [])
                previous_observation = dict(steps[step_idx - 1][agent_idx].get("observation") or {})
                if not previous_observation.get("planets"):
                    continue
                previous_observation.setdefault("step", fallback_step)
                state = parse_observation(previous_observation)
                action_by_source: dict[int, list[list[float | int]]] = defaultdict(list)
                for action in actions:
                    action_by_source[int(action[0])].append(action)

                batch_rows = build_rows_for_state(
                    state=state,
                    cfg=cfg,
                    action_by_source=action_by_source,
                    angle_threshold=float(args.angle_threshold),
                    no_op_ratio=float(args.no_op_ratio),
                    rng=rng,
                    stats=stats,
                )
                for row in batch_rows:
                    (
                        self_feat,
                        cand_feat,
                        global_feat,
                        mask,
                        ship_bucket_mask,
                        bucket_feat,
                        label,
                        bucket_label,
                    ) = row
                    self_rows.append(self_feat)
                    candidate_rows.append(cand_feat)
                    global_rows.append(global_feat)
                    mask_rows.append(mask)
                    ship_bucket_mask_rows.append(ship_bucket_mask)
                    bucket_feat_rows.append(bucket_feat)
                    labels.append(label)
                    bucket_labels.append(bucket_label)
                    if args.max_rows > 0 and len(labels) >= args.max_rows:
                        write_output(
                            args.output,
                            self_rows,
                            candidate_rows,
                            global_rows,
                            mask_rows,
                            ship_bucket_mask_rows,
                            bucket_feat_rows,
                            labels,
                            bucket_labels,
                            stats,
                            args,
                            cfg,
                        )
                        return

    write_output(
        args.output,
        self_rows,
        candidate_rows,
        global_rows,
        mask_rows,
        ship_bucket_mask_rows,
        bucket_feat_rows,
        labels,
        bucket_labels,
        stats,
        args,
        cfg,
    )


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (RL_ROOT / candidate).resolve()


def valid_actions(actions: Any) -> list[list[float | int]]:
    valid: list[list[float | int]] = []
    if not isinstance(actions, list):
        return valid
    for action in actions:
        if isinstance(action, list) and len(action) >= 3:
            valid.append([int(action[0]), float(action[1]), int(action[2])])
    return valid


def build_rows_for_state(
    *,
    state: GameState,
    cfg: Any,
    action_by_source: dict[int, list[list[float | int]]],
    angle_threshold: float,
    no_op_ratio: float,
    rng: random.Random,
    stats: Counter[str],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    my_planets = sorted((planet for planet in state.planets if planet.owner == state.player), key=lambda p: p.id)
    if not my_planets:
        return []
    planet_by_id = {p.id: p for p in state.planets}
    global_feat = build_global_features(state, cfg.env)
    cache = build_feature_cache(state, cfg.env)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    for source in my_planets:
        source_actions = action_by_source.get(source.id, [])
        if len(source_actions) > 1:
            stats["duplicate_source_actions"] += len(source_actions) - 1
            source_actions = source_actions[:1]

        cand_feat, mask, bucket_mask, _ship_counts, bucket_counts, candidate_ids, _target_angles = (
            build_candidate_features(source, state, cfg.env, cache)
        )
        bucket_feat = build_bucket_features(source, state, bucket_counts, bucket_mask, cfg.env)

        target_idx = 0
        bucket_idx = 0

        if source_actions:
            stats["action_rows_seen"] += 1
            from_planet_id = int(source_actions[0][0])
            action_angle = float(source_actions[0][1])
            num_ships = int(source_actions[0][2])

            if from_planet_id != source.id:
                stats["action_rows_skipped_source_mismatch"] += 1
                continue

            to_planet_id = resolve_action_target_id(
                source,
                action_angle,
                state,
                threshold=angle_threshold,
            )
            if to_planet_id is None:
                stats["action_rows_skipped_unresolved_target"] += 1
                continue

            matched_idx = candidate_index_for_target(to_planet_id, candidate_ids, mask)
            if matched_idx is None:
                # Planet exists on board but mission invalid (e.g. sun block) — still label slot.
                slot = planet_slot_for_id(to_planet_id)
                if slot >= len(mask) or candidate_ids[slot] != to_planet_id:
                    stats["action_rows_skipped_not_in_candidates"] += 1
                    continue
                matched_idx = slot

            tgt = planet_by_id.get(to_planet_id)
            if tgt is None:
                stats["action_rows_skipped_missing_planet"] += 1
                continue

            target_idx = int(matched_idx)
            bucket_idx = bucket_index_for_ships(source, tgt, state, num_ships, cfg.env)
            stats["action_rows_matched"] += 1
        else:
            stats["no_op_rows_seen"] += 1
            if no_op_ratio <= 0.0 or rng.random() > no_op_ratio:
                stats["no_op_rows_dropped"] += 1
                continue
            stats["no_op_rows_kept"] += 1

        rows.append(
            (
                build_self_features(source, state, cfg.env, cache),
                cand_feat,
                global_feat,
                mask,
                bucket_mask,
                bucket_feat,
                target_idx,
                bucket_idx,
            )
        )
    return rows


def write_output(
    output_path: str | Path,
    self_rows: list[np.ndarray],
    candidate_rows: list[np.ndarray],
    global_rows: list[np.ndarray],
    mask_rows: list[np.ndarray],
    ship_bucket_mask_rows: list[np.ndarray],
    bucket_feat_rows: list[np.ndarray],
    labels: list[int],
    bucket_labels: list[int],
    stats: Counter[str],
    args: argparse.Namespace,
    cfg: Any,
) -> None:
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(labels)
    action_seen = int(stats.get("action_rows_seen", 0))
    skipped = (
        int(stats.get("action_rows_skipped_not_in_candidates", 0))
        + int(stats.get("action_rows_skipped_unresolved_target", 0))
        + int(stats.get("action_rows_skipped_source_mismatch", 0))
        + int(stats.get("action_rows_skipped_missing_planet", 0))
    )
    skip_rate = skipped / action_seen if action_seen > 0 else 0.0

    slots = target_slot_count(cfg.env)
    buckets = int(cfg.env.ship_bucket_count)
    np.savez_compressed(
        output,
        self_features=np.asarray(self_rows, dtype=np.float32).reshape(count, self_feature_dim()),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32).reshape(count, -1, candidate_feature_dim()),
        global_features=np.asarray(global_rows, dtype=np.float32).reshape(count, -1),
        candidate_mask=np.asarray(mask_rows, dtype=bool).reshape(count, -1),
        ship_bucket_mask=np.asarray(ship_bucket_mask_rows, dtype=bool).reshape(count, slots, buckets),
        bucket_features=np.asarray(bucket_feat_rows, dtype=np.float32).reshape(
            count, slots, buckets, bucket_feature_dim()
        ),
        target_index=np.asarray(labels, dtype=np.int64),
        ship_bucket_index=np.asarray(bucket_labels, dtype=np.int64),
    )
    metadata = {
        "rows": count,
        "stats": dict(stats),
        "skip_rate": skip_rate,
        "args": vars(args),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if skip_rate > 0.30:
        print(
            f"WARNING: action skip_rate={skip_rate:.1%} exceeds 30% — "
            "candidate set may be too narrow for replay coverage."
        )


if __name__ == "__main__":
    main()
