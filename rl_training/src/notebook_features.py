
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .game_types import GameState, PlanetState
from .geometry import BOARD_CENTER, estimate_travel_time, fleet_speed

# orbit-wars-agent-ow-proto-passed-1-000.ipynb
PROTO_FORMULA_DIST = 100.0
PROTO_FORMULA_PROD_MULT = 15.0
PROTO_FORMULA_ENEMY_BONUS_MULT = 10.0
PROTO_FORMULA_TOTAL_SHIPS_PERCENT = 0.7
PROTO_COMET_PENALTY = 40.0
PROTO_MIN_COOP_SHIPS = 20

# lb-highest-1000-search-learned-value-function.ipynb
LB_CENTRALITY_SCALE = 60.0

# hellburner-x2.ipynb
HELLBURNER_MAX_TRAVEL = 88.0


@dataclass(slots=True)
class FeatureCache:
    under_attack_ids: set[int] = field(default_factory=set)
    planet_centrality: dict[int, float] = field(default_factory=dict)
    owner_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    best_enemy_owner: int | None = None
    n_players: int = 2


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-9 else 0.0


def planet_centrality(planet: PlanetState) -> float:
    dist_center = math.hypot(planet.x - BOARD_CENTER[0], planet.y - BOARD_CENTER[1])
    return max(0.0, LB_CENTRALITY_SCALE - dist_center) / LB_CENTRALITY_SCALE


def collides_segment_circle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(cx - x1, cy - y1) <= radius
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(cx - closest_x, cy - closest_y) <= radius


def compute_under_attack_ids(state: GameState, player: int, *, horizon: int = 60) -> set[int]:
    """Proto-style fleet ray collision against owned planets."""
    attacked: set[int] = set()
    owned = [planet for planet in state.planets if planet.owner == player]
    if not owned:
        return attacked
    for fleet in state.fleets:
        if fleet.owner == player:
            continue
        speed = fleet_speed(fleet.ships)
        prev_x, prev_y = fleet.x, fleet.y
        for tick in range(1, horizon + 1):
            next_x = prev_x + math.cos(fleet.angle) * speed
            next_y = prev_y + math.sin(fleet.angle) * speed
            for planet in owned:
                if planet.id in attacked:
                    continue
                if collides_segment_circle(prev_x, prev_y, next_x, next_y, planet.x, planet.y, planet.radius):
                    attacked.add(planet.id)
            prev_x, prev_y = next_x, next_y
    return attacked


def aggregate_owner_stats(state: GameState) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for planet in state.planets:
        owner = int(planet.owner)
        if owner == -1:
            continue
        row = stats.setdefault(
            owner,
            {"planets": 0.0, "ships": 0.0, "production": 0.0, "centrality_sum": 0.0, "centrality_count": 0.0},
        )
        row["planets"] += 1.0
        row["ships"] += float(planet.ships)
        row["production"] += float(planet.production)
        row["centrality_sum"] += max(0.0, LB_CENTRALITY_SCALE - math.hypot(planet.x - BOARD_CENTER[0], planet.y - BOARD_CENTER[1]))
        row["centrality_count"] += 1.0
    for fleet in state.fleets:
        owner = int(fleet.owner)
        if owner == -1:
            continue
        row = stats.setdefault(owner, {"planets": 0.0, "ships": 0.0, "production": 0.0, "centrality_sum": 0.0, "centrality_count": 0.0})
        row["ships"] += float(fleet.ships)
    return stats


def best_enemy_owner(state: GameState, player: int, owner_stats: dict[int, dict[str, float]]) -> int | None:
    enemies = [owner for owner in owner_stats if owner not in {-1, player}]
    if not enemies:
        return None

    def enemy_score(owner: int) -> float:
        row = owner_stats[owner]
        return row["ships"] + 100.0 * row["planets"]

    return max(enemies, key=enemy_score)


def build_feature_cache(state: GameState, env_cfg: EnvConfig) -> FeatureCache:
    owner_stats = aggregate_owner_stats(state)
    centrality = {planet.id: planet_centrality(planet) for planet in state.planets}
    return FeatureCache(
        under_attack_ids=compute_under_attack_ids(state, state.player),
        planet_centrality=centrality,
        owner_stats=owner_stats,
        best_enemy_owner=best_enemy_owner(state, state.player, owner_stats),
        n_players=2,
    )


def lb_global_feature_dim() -> int:
    return 16


def build_lb_global_features(state: GameState, env_cfg: EnvConfig, cache: FeatureCache) -> np.ndarray:
    """16-dim value-state vector from lb-highest-1000-search-learned-value-function.ipynb."""
    player = state.player
    owner_stats = cache.owner_stats
    my = owner_stats.get(player, {"planets": 0.0, "ships": 0.0, "production": 0.0, "centrality_sum": 0.0, "centrality_count": 0.0})
    best_owner = cache.best_enemy_owner
    if best_owner is None:
        enemy = {"planets": 0.0, "ships": 0.0, "production": 0.0, "centrality_sum": 0.0, "centrality_count": 0.0}
    else:
        enemy = owner_stats[best_owner]

    my_planets = my["planets"]
    my_ships = my["ships"]
    my_prod = my["production"]
    my_centrality = safe_div(my["centrality_sum"], my["centrality_count"])
    enemy_planets = enemy["planets"]
    enemy_ships = enemy["ships"]
    enemy_prod = enemy["production"]
    enemy_centrality = safe_div(enemy["centrality_sum"], enemy["centrality_count"])

    on_planet_ships = sum(float(planet.ships) for planet in state.planets)
    total_planets = float(len(state.planets))
    total_prod = sum(float(planet.production) for planet in state.planets if planet.owner != -1)
    my_fleet_ships = sum(float(fleet.ships) for fleet in state.fleets if fleet.owner == player)
    in_flight_fraction = safe_div(my_fleet_ships, max(1.0, my_ships))

    return np.asarray(
        [
            state.step / max(1.0, float(env_cfg.episode_steps)),
            cache.n_players / 4.0,
            safe_div(my_planets, total_planets),
            safe_div(enemy_planets, total_planets),
            safe_div(my_ships, on_planet_ships + sum(float(f.ships) for f in state.fleets)),
            safe_div(enemy_ships, on_planet_ships + sum(float(f.ships) for f in state.fleets)),
            safe_div(my_prod, total_prod),
            safe_div(enemy_prod, total_prod),
            my_centrality / LB_CENTRALITY_SCALE,
            enemy_centrality / LB_CENTRALITY_SCALE,
            in_flight_fraction,
            safe_div(my_ships - enemy_ships, on_planet_ships + sum(float(f.ships) for f in state.fleets)),
            safe_div(my_planets - enemy_planets, total_planets),
            safe_div(my_prod - enemy_prod, total_prod),
            1.0 if cache.n_players == 2 else 0.0,
            1.0 if cache.n_players == 4 else 0.0,
        ],
        dtype=np.float32,
    )


def proto_move_score(
    src: PlanetState,
    tgt: PlanetState,
    state: GameState,
    *,
    min_ships: int | None = None,
) -> tuple[float, float, float]:
    """Returns (score, eta, capture_ships_at_arrival) from Proto notebook formula."""
    dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
    capture_min = max(1, int(min_ships if min_ships is not None else tgt.ships + 1))
    eta = estimate_travel_time(src, tgt, capture_min, state.angular_velocity)
    if tgt.owner not in {-1, state.player}:
        enemy_produced = eta * float(tgt.production)
        enemy_bonus = float(tgt.production)
    else:
        enemy_produced = 0.0
        enemy_bonus = 0.0
    total_ships = float(capture_min) + enemy_produced
    score = (
        (PROTO_FORMULA_DIST - dist)
        + (PROTO_FORMULA_PROD_MULT * float(tgt.production))
        + (PROTO_FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        - (PROTO_FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2.0 * eta)
    )
    if tgt.id in state.comet_planet_ids:
        score -= PROTO_COMET_PENALTY
    return score, eta, total_ships


def hellburner_reachable(src: PlanetState, tgt: PlanetState, ships: int, state: GameState) -> bool:
    travel = estimate_travel_time(src, tgt, max(1, ships), state.angular_velocity)
    return math.isfinite(travel) and travel <= HELLBURNER_MAX_TRAVEL


def count_reachable_sources(target: PlanetState, state: GameState, player: int) -> int:
    count = 0
    for planet in state.planets:
        if planet.owner != player or planet.id == target.id:
            continue
        if hellburner_reachable(planet, target, max(5, int(planet.ships * 0.5)), state):
            count += 1
    return count


def target_losing_without_help(tgt: PlanetState, state: GameState, cache: FeatureCache) -> float:
    enemy_incoming = 0.0
    friendly_incoming = 0.0
    for fleet in state.fleets:
        dx = tgt.x - fleet.x
        dy = tgt.y - fleet.y
        if math.hypot(dx, dy) > 40.0:
            continue
        bearing = math.atan2(dy, dx)
        delta = abs((bearing - fleet.angle + math.pi) % (2.0 * math.pi) - math.pi)
        if delta > 0.25:
            continue
        if fleet.owner == state.player:
            friendly_incoming += float(fleet.ships)
        elif fleet.owner != -1:
            enemy_incoming += float(fleet.ships)
    deficit = enemy_incoming - float(tgt.ships) - friendly_incoming
    return 1.0 if deficit > 0.0 else 0.0


def needs_coop_attack(src: PlanetState, tgt: PlanetState, state: GameState) -> float:
    if tgt.owner == state.player or tgt.owner == -1:
        return 0.0
    surplus = max(0, int(src.ships) - max(8, int(round(0.2 * src.ships))))
    base = max(int(tgt.ships) + 1, 20)
    return 1.0 if tgt.ships >= PROTO_MIN_COOP_SHIPS and surplus < base else 0.0
