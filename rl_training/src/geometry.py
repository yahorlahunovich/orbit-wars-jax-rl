
from __future__ import annotations

import math

from .game_types import PlanetState

BOARD_CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0
MAX_SHIP_SPEED = 6.0


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fleet_speed(ships: int, max_speed: float = MAX_SHIP_SPEED) -> float:
    ships = max(1, int(ships))
    scaled = (math.log(ships) / math.log(1000.0)) ** 1.5 if ships > 1 else 0.0
    return 1.0 + (max_speed - 1.0) * min(1.0, scaled)


def is_orbiting_planet(planet: PlanetState) -> bool:
    orbit_radius = distance_xy((planet.x, planet.y), BOARD_CENTER)
    return orbit_radius + planet.radius < ROTATION_RADIUS_LIMIT


def predict_planet_position(
    planet: PlanetState,
    turns_ahead: float,
    angular_velocity: float,
) -> tuple[float, float]:
    if not is_orbiting_planet(planet):
        return (planet.x, planet.y)
    cx, cy = BOARD_CENTER
    dx = planet.x - cx
    dy = planet.y - cy
    radians = angular_velocity * turns_ahead
    c = math.cos(radians)
    s = math.sin(radians)
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


def estimate_travel_time(
    source: PlanetState,
    target: PlanetState,
    ships: int,
    angular_velocity: float,
    *,
    iterations: int = 4,
) -> float:
    speed = fleet_speed(max(1, ships))
    source_xy = (source.x, source.y)
    target_xy = (target.x, target.y)
    travel_time = distance_xy(source_xy, target_xy) / speed
    for _ in range(iterations):
        target_xy = predict_planet_position(target, travel_time, angular_velocity)
        travel_time = distance_xy(source_xy, target_xy) / speed
    return travel_time
