from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import setup_rl_script_paths

REPO_ROOT, RL_ROOT = setup_rl_script_paths()

from build_bc_dataset import build_rows_for_state, resolve_path, write_output  # noqa: E402
from src.config import load_train_config  # noqa: E402

CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0

PARQUET_TABLES = (
    "episodes.parquet",
    "player_episodes.parquet",
    "actions.parquet",
    "episode_planets.parquet",
    "planet_state.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BC .npz from Orbit Wars replay parquet tables (not thousands of JSON files)."
    )
    parser.add_argument(
        "--data-dir",
        default="/media/yahor/ADATA SE880/datasets/orbit-wars/parquet_from_json_top_players",
        help="Directory containing episodes.parquet, actions.parquet, etc.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts/bc/top_players_parquet_bc.npz")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--angle-threshold", type=float, default=0.35)
    parser.add_argument("--no-op-ratio", type=float, default=0.05)
    parser.add_argument("--include-losers", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def validate_parquet_dir(data_dir: Path) -> None:
    missing = [name for name in PARQUET_TABLES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing parquet tables in {data_dir}: {missing}. "
            "Run json_replays_to_parquet.py locally first, then upload the folder to Kaggle."
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    cfg = load_train_config(resolve_path(args.config))
    root = Path(args.data_dir)
    validate_parquet_dir(root)

    episodes = pd.read_parquet(root / "episodes.parquet")
    players = pd.read_parquet(root / "player_episodes.parquet")
    actions = pd.read_parquet(root / "actions.parquet")
    planets = pd.read_parquet(root / "episode_planets.parquet")
    planet_state = pd.read_parquet(root / "planet_state.parquet")

    selected_players = players if args.include_losers else players[players["is_winner"] == 1]
    actions = actions.merge(
        selected_players[["episode_id", "slot"]],
        on=["episode_id", "slot"],
        how="inner",
    )
    actions = actions[actions["tick"] > 0].copy()
    actions["prev_tick"] = actions["tick"].astype(np.int16) - 1
    if args.max_episodes > 0:
        keep_eps = set(sorted(actions["episode_id"].unique())[: args.max_episodes])
        actions = actions[actions["episode_id"].isin(keep_eps)]

    episode_meta = episodes.set_index("episode_id").to_dict("index")
    planets_by_episode = {int(ep): df.copy() for ep, df in planets.groupby("episode_id", sort=False)}
    states_by_episode = {int(ep): df.copy() for ep, df in planet_state.groupby("episode_id", sort=False)}

    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    ship_bucket_mask_rows: list[np.ndarray] = []
    bucket_feat_rows: list[np.ndarray] = []
    labels: list[int] = []
    bucket_labels: list[int] = []
    stats: Counter[str] = Counter()

    for episode_id, ep_actions in actions.groupby("episode_id", sort=False):
        episode_id = int(episode_id)
        meta = episode_meta.get(episode_id)
        topo = planets_by_episode.get(episode_id)
        ep_states = states_by_episode.get(episode_id)
        if meta is None or topo is None or ep_states is None:
            stats["missing_episode_tables"] += 1
            continue
        state_by_tick = {int(tick): df for tick, df in ep_states.groupby("tick", sort=False)}
        topo_records = [row for row in topo.itertuples(index=False)]
        for (_tick, prev_tick, slot), turn_actions in ep_actions.groupby(["tick", "prev_tick", "slot"], sort=False):
            prev_tick = int(prev_tick)
            slot = int(slot)
            tick_state = state_by_tick.get(prev_tick)
            if tick_state is None:
                stats["missing_prev_tick_state"] += 1
                continue
            state = build_game_state(
                episode_id=episode_id,
                tick=prev_tick,
                slot=slot,
                angular_velocity=float(meta["angular_velocity"]),
                topo_records=topo_records,
                tick_state=tick_state,
            )
            if state is None:
                stats["bad_state"] += 1
                continue
            action_by_source: dict[int, list[list[float | int]]] = defaultdict(list)
            for action in turn_actions.itertuples(index=False):
                action_by_source[int(action.src_planet_id)].append(
                    [int(action.src_planet_id), float(action.angle), int(action.n_ships)]
                )
            rows = build_rows_for_state(
                state=state,
                cfg=cfg,
                action_by_source=action_by_source,
                angle_threshold=float(args.angle_threshold),
                no_op_ratio=float(args.no_op_ratio),
                rng=rng,
                stats=stats,
            )
            for row in rows:
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


def build_game_state(
    *,
    episode_id: int,
    tick: int,
    slot: int,
    angular_velocity: float,
    topo_records: list[Any],
    tick_state: pd.DataFrame,
) -> Any:
    from src.game_types import GameState, PlanetState

    state_by_planet = {
        int(row.planet_id): (int(row.owner), int(row.ships))
        for row in tick_state.itertuples(index=False)
    }
    planets: list[PlanetState] = []
    comet_ids: set[int] = set()
    for row in topo_records:
        planet_id = int(row.planet_id)
        owner_ships = state_by_planet.get(planet_id)
        if owner_ships is None:
            continue
        owner, ships = owner_ships
        x, y = planet_position(row, angular_velocity, tick)
        planets.append(
            PlanetState(
                id=planet_id,
                owner=owner,
                x=float(x),
                y=float(y),
                radius=float(row.radius),
                ships=ships,
                production=int(row.production),
            )
        )
        if bool(row.is_comet):
            comet_ids.add(planet_id)
    if not planets:
        return None
    return GameState(
        step=int(tick),
        player=int(slot),
        planets=planets,
        fleets=[],
        angular_velocity=float(angular_velocity),
        comet_planet_ids=comet_ids,
    )


def planet_position(row: Any, angular_velocity: float, tick: int) -> tuple[float, float]:
    x0 = float(row.initial_x)
    y0 = float(row.initial_y)
    radius = float(row.radius)
    orbit_radius = float(row.orbit_radius)
    if bool(row.is_static) or bool(row.is_comet) or orbit_radius + radius >= ROTATION_RADIUS_LIMIT:
        return x0, y0
    initial_angle = math.atan2(y0 - CENTER, x0 - CENTER)
    current_angle = initial_angle + float(angular_velocity) * int(tick)
    return (
        CENTER + orbit_radius * math.cos(current_angle),
        CENTER + orbit_radius * math.sin(current_angle),
    )


if __name__ == "__main__":
    main()
