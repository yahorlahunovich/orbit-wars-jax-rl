from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.game import GameState, Planet
from src.geometry import estimate_intercept, path_hits_sun

DEFAULT_CONFIG = {
    "min_defense": 12,
    "launch_fraction": 0.65,
    "production_weight": 32.0,
    "distance_weight": 0.45,
    "ships_weight": 1.0,
    "enemy_bonus": 8.0,
    "comet_bonus": 6.0,
    "max_targets_per_source": 8,
    "safety_margin": 2,
    "avoid_sun_margin": 0.4,
}


def load_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "bot_config.json"
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        return merged
    except Exception:
        return dict(DEFAULT_CONFIG)


def future_target_ships(target: Planet, travel_time: float) -> int:
    if target.owner == -1:
        return target.ships
    return int(target.ships + target.production * max(0.0, travel_time))


def score_target(
    state: GameState,
    source: Planet,
    target: Planet,
    send_ships: int,
    cfg: dict[str, Any],
) -> tuple[float, float, int, float] | None:
    angle, travel_time, dist, _ = estimate_intercept(
        source,
        target,
        send_ships,
        state.angular_velocity,
    )
    if path_hits_sun((source.x, source.y), (target.x, target.y), cfg["avoid_sun_margin"]):
        return None

    needed = future_target_ships(target, travel_time) + int(cfg["safety_margin"])
    if send_ships < needed:
        return None

    remaining_turns = max(1.0, 500.0 - state.step - travel_time)
    production_value = target.production * cfg["production_weight"]
    payoff_value = target.production * min(remaining_turns, 120.0) * 0.12
    cost = needed * cfg["ships_weight"] + dist * cfg["distance_weight"]
    ownership_bonus = cfg["enemy_bonus"] if target.owner not in (-1, state.player) else 0.0
    comet_bonus = cfg["comet_bonus"] if target.id in state.comet_planet_ids else 0.0
    score = production_value + payoff_value + ownership_bonus + comet_bonus - cost
    return score, angle, needed, travel_time


def choose_expansion_moves(state: GameState, cfg: dict[str, Any]) -> list[list[int | float]]:
    moves: list[list[int | float]] = []
    my_planets = [p for p in state.planets if p.owner == state.player]
    targets = [p for p in state.planets if p.owner != state.player]

    if not my_planets or not targets:
        return moves

    my_planets.sort(key=lambda p: p.ships, reverse=True)

    targeted: set[int] = set()
    for source in my_planets:
        reserve = int(cfg["min_defense"] + source.production * 2)
        available = max(0, source.ships - reserve)
        send_cap = int(available * cfg["launch_fraction"])
        if send_cap <= 1:
            continue

        candidates = sorted(
            targets,
            key=lambda t: abs(t.x - source.x) + abs(t.y - source.y),
        )[: int(cfg["max_targets_per_source"])]

        best: tuple[float, float, int, int] | None = None
        for target in candidates:
            if target.id in targeted:
                continue
            result = score_target(state, source, target, send_cap, cfg)
            if result is None:
                continue
            score, angle, needed, _ = result
            if best is None or score > best[0]:
                best = (score, angle, needed, target.id)

        if best is None:
            continue

        score, angle, needed, target_id = best
        if score <= -10.0:
            continue
        moves.append([source.id, float(angle), int(min(needed, send_cap))])
        targeted.add(target_id)

    return moves


def decide_moves(state: GameState) -> list[list[int | float]]:
    cfg = load_config()
    return choose_expansion_moves(state, cfg)
