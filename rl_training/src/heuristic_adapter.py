
from __future__ import annotations

import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

_PREPARED = False
_HEURISTIC_ROOT: Path | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def heuristic_version_root(version: str = "kaggle700_current_heuristic") -> Path:
    root = repo_root() / "versions" / version
    if not (root / "src" / "bot.py").exists():
        raise FileNotFoundError(f"Heuristic opponent not found: {root}")
    return root


def prepare_heuristic_opponent(version: str = "kaggle700_current_heuristic") -> Path:
    global _PREPARED, _HEURISTIC_ROOT
    root = heuristic_version_root(version)
    if _PREPARED and _HEURISTIC_ROOT == root:
        return root
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _HEURISTIC_ROOT = root
    _PREPARED = True
    return root


def heuristic_agent(obs: Any, *, version: str = "kaggle700_current_heuristic") -> list[list[int | float]]:
    try:
        prepare_heuristic_opponent(version)
        bot = import_module("src.bot")
        return list(bot.agent(obs))
    except Exception:
        return []


def heuristic_plan_for_target(
    state: Any,
    source_id: int,
    target_id: int,
    *,
    version: str = "kaggle700_current_heuristic",
) -> tuple[float, int] | None:
    try:
        prepare_heuristic_opponent(version)
        game = import_module("src.game")
        strategy = import_module("src.strategy")

        h_state = _to_heuristic_state(state, game)
        planets_by_id = {int(planet.id): planet for planet in h_state.planets}
        source = planets_by_id.get(int(source_id))
        target = planets_by_id.get(int(target_id))
        if source is None or target is None or target.owner == h_state.player:
            return None

        base = strategy.load_config()
        cfg = strategy.effective_cfg(h_state, base)
        arrivals = strategy.build_arrivals_by_planet(h_state, float(cfg["threat_hit_slop"]))
        threats = strategy.aggregate_threats_by_planet(h_state, cfg)
        buf = strategy.per_planet_ship_buffer(h_state.step, source.production, cfg)
        avail = int(source.ships) - int(buf)
        if avail <= 0:
            return None

        obstacles = strategy.obstacles_for_path(h_state, source.id, target.id, cfg)
        got = strategy.find_capture_launch(
            h_state,
            source,
            target,
            avail,
            cfg,
            arrivals.get(target.id, []),
            obstacles,
            target.owner not in (-1, h_state.player),
        )
        if got is None:
            return None
        angle, send, _travel_time, _pred_xy = got
        if not strategy.source_survives_after_launch(source, int(send), threats.get(source.id, []), cfg):
            return None
        return float(angle), int(send)
    except Exception:
        return None


def _to_heuristic_state(state: Any, game: Any) -> Any:
    planets = [
        game.Planet(
            id=int(planet.id),
            owner=int(planet.owner),
            x=float(planet.x),
            y=float(planet.y),
            radius=float(planet.radius),
            ships=int(planet.ships),
            production=int(planet.production),
        )
        for planet in getattr(state, "planets", [])
    ]
    fleets = [
        game.Fleet(
            id=int(fleet.id),
            owner=int(fleet.owner),
            x=float(fleet.x),
            y=float(fleet.y),
            angle=float(fleet.angle),
            from_planet_id=int(fleet.from_planet_id),
            ships=int(fleet.ships),
        )
        for fleet in getattr(state, "fleets", [])
    ]
    return game.GameState(
        player=int(getattr(state, "player", 0)),
        planets=planets,
        fleets=fleets,
        comet_planet_ids={int(item) for item in getattr(state, "comet_planet_ids", set())},
        angular_velocity=float(getattr(state, "angular_velocity", 0.0)),
        step=int(getattr(state, "step", 0)),
    )
