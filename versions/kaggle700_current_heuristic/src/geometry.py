from __future__ import annotations

import math

from src.constants import CENTER, MAX_SHIP_SPEED, ROTATION_RADIUS_LIMIT, SUN_RADIUS
from src.game import Fleet, Planet


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def fleet_speed(ships: int, max_speed: float = MAX_SHIP_SPEED) -> float:
    ships = max(1, int(ships))
    scaled = (math.log(ships) / math.log(1000.0)) ** 1.5 if ships > 1 else 0.0
    return 1.0 + (max_speed - 1.0) * min(1.0, scaled)


def point_segment_distance(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return distance_xy(point, a)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return distance_xy(point, closest)


def segment_intersects_circle(
    a: tuple[float, float],
    b: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    return point_segment_distance(center, a, b) <= radius


def path_hits_sun(
    a: tuple[float, float],
    b: tuple[float, float],
    margin: float = 0.0,
) -> bool:
    return segment_intersects_circle(a, b, CENTER, SUN_RADIUS + margin)


def segment_clear_of_circles(
    a: tuple[float, float],
    b: tuple[float, float],
    circles: list[tuple[tuple[float, float], float]],
) -> bool:
    """True if segment ab does not intersect any circle (each circle = (center, radius))."""
    for center, radius in circles:
        if point_segment_distance(center, a, b) <= radius:
            return False
    return True


def rotate_around_center(x: float, y: float, radians: float) -> tuple[float, float]:
    cx, cy = CENTER
    dx = x - cx
    dy = y - cy
    c = math.cos(radians)
    s = math.sin(radians)
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


def is_orbiting_planet(planet: Planet) -> bool:
    orbit_radius = distance_xy((planet.x, planet.y), CENTER)
    return orbit_radius + planet.radius < ROTATION_RADIUS_LIMIT


def predict_planet_position(
    planet: Planet,
    turns_ahead: float,
    angular_velocity: float,
) -> tuple[float, float]:
    if not is_orbiting_planet(planet):
        return (planet.x, planet.y)
    return rotate_around_center(planet.x, planet.y, angular_velocity * turns_ahead)


def estimate_intercept(
    source: Planet,
    target: Planet,
    ships: int,
    angular_velocity: float,
    iterations: int = 4,
) -> tuple[float, float, float, float, tuple[float, float]]:
    speed = fleet_speed(ships)
    source_xy = (source.x, source.y)
    target_xy = (target.x, target.y)
    travel_time = distance_xy(source_xy, target_xy) / speed
    for _ in range(iterations):
        target_xy = predict_planet_position(target, travel_time, angular_velocity)
        travel_time = distance_xy(source_xy, target_xy) / speed
    angle = angle_between(source_xy, target_xy)
    dist = distance_xy(source_xy, target_xy)
    return angle, travel_time, dist, speed, target_xy


def fleet_ray_closest_to_point(
    fleet: Fleet,
    px: float,
    py: float,
) -> tuple[float, float]:
    """Forward ray from fleet: position + t * speed * (cos a, sin a), t >= 0. Returns (t_close, distance)."""
    sp = fleet_speed(fleet.ships)
    ux = math.cos(fleet.angle)
    uy = math.sin(fleet.angle)
    rx = px - fleet.x
    ry = py - fleet.y
    dot_ru = rx * ux + ry * uy
    if dot_ru <= 0.0:
        return 0.0, distance_xy((fleet.x, fleet.y), (px, py))
    t_close = dot_ru / sp
    cx = fleet.x + sp * ux * t_close
    cy = fleet.y + sp * uy * t_close
    return t_close, distance_xy((px, py), (cx, cy))
