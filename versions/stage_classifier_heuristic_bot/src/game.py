from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int

    @classmethod
    def from_raw(cls, raw: list[Any]) -> "Planet":
        return cls(
            id=int(raw[0]),
            owner=int(raw[1]),
            x=float(raw[2]),
            y=float(raw[3]),
            radius=float(raw[4]),
            ships=int(raw[5]),
            production=int(raw[6]),
        )


@dataclass(frozen=True)
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int

    @classmethod
    def from_raw(cls, raw: list[Any]) -> "Fleet":
        return cls(
            id=int(raw[0]),
            owner=int(raw[1]),
            x=float(raw[2]),
            y=float(raw[3]),
            angle=float(raw[4]),
            from_planet_id=int(raw[5]),
            ships=int(raw[6]),
        )


@dataclass(frozen=True)
class GameState:
    player: int
    planets: list[Planet]
    fleets: list[Fleet]
    comet_planet_ids: set[int]
    angular_velocity: float
    step: int


def obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse_state(obs: Any) -> GameState:
    planets = [Planet.from_raw(p) for p in obs_get(obs, "planets", [])]
    fleets = [Fleet.from_raw(f) for f in obs_get(obs, "fleets", [])]
    comet_planet_ids = {int(x) for x in obs_get(obs, "comet_planet_ids", [])}
    return GameState(
        player=int(obs_get(obs, "player", 0)),
        planets=planets,
        fleets=fleets,
        comet_planet_ids=comet_planet_ids,
        angular_velocity=float(obs_get(obs, "angular_velocity", 0.0)),
        step=int(obs_get(obs, "step", obs_get(obs, "turn", 0))),
    )
