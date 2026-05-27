
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .game_types import GameState, PlanetState, parse_observation
from .geometry import estimate_travel_time
from .heuristic_adapter import heuristic_plan_for_target
from .notebook_features import (
    FeatureCache,
    build_feature_cache,
    build_lb_global_features,
    count_reachable_sources,
    lb_global_feature_dim,
    needs_coop_attack,
    planet_centrality,
    proto_move_score,
    target_losing_without_help,
)

BOARD_CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS = 10.0
PLANET_LAUNCH_RADIUS_OFFSET = 0.1

# Ship buckets: 25%, 50%, 75%, 100% of surplus, plus exact mission size.
SHIP_BUCKET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
EXACT_BUCKET_INDEX = 4


@dataclass(slots=True)
class DecisionContext:
    env_index: int
    source_id: int
    candidate_ids: list[int]
    candidate_mask: np.ndarray
    ship_bucket_mask: np.ndarray
    ship_counts: list[int]
    ship_count_buckets: list[list[int]]
    target_angles: list[float]


@dataclass(slots=True)
class TurnBatch:
    self_features: np.ndarray
    candidate_features: np.ndarray
    global_features: np.ndarray
    candidate_mask: np.ndarray
    ship_bucket_mask: np.ndarray
    bucket_features: np.ndarray
    contexts: list[DecisionContext]
    state: GameState


def self_feature_dim() -> int:
    return 22


def candidate_feature_dim() -> int:
    return 28


def global_feature_dim() -> int:
    return lb_global_feature_dim()


def bucket_feature_dim() -> int:
    return 4


def target_slot_count(env_cfg: EnvConfig) -> int:
    """Slot 0 = no-op; slot planet_id + 1 = that planet."""
    return int(env_cfg.max_planets) + 1


def planet_slot_for_id(planet_id: int) -> int:
    return int(planet_id) + 1


def planet_id_from_slot(slot: int) -> int | None:
    if slot <= 0:
        return None
    return int(slot) - 1


def ship_bucket_count(env_cfg: EnvConfig) -> int:
    return int(env_cfg.ship_bucket_count)


def encode_turn(
    observation: Any,
    env_cfg: EnvConfig,
    *,
    env_index: int = 0,
) -> TurnBatch:
    state = observation if isinstance(observation, GameState) else parse_observation(observation)
    slots = target_slot_count(env_cfg)
    buckets = ship_bucket_count(env_cfg)
    my_planets = sorted((planet for planet in state.planets if planet.owner == state.player), key=lambda planet: planet.id)
    if not my_planets:
        return TurnBatch(
            self_features=np.zeros((0, self_feature_dim()), dtype=np.float32),
            candidate_features=np.zeros((0, slots, candidate_feature_dim()), dtype=np.float32),
            global_features=np.zeros((0, global_feature_dim()), dtype=np.float32),
            candidate_mask=np.zeros((0, slots), dtype=bool),
            ship_bucket_mask=np.zeros((0, slots, buckets), dtype=bool),
            bucket_features=np.zeros((0, slots, buckets, bucket_feature_dim()), dtype=np.float32),
            contexts=[],
            state=state,
        )

    global_feat = build_global_features(state, env_cfg)
    cache = build_feature_cache(state, env_cfg)
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    ship_bucket_masks: list[np.ndarray] = []
    bucket_feat_rows: list[np.ndarray] = []
    contexts: list[DecisionContext] = []

    for src in my_planets:
        cand_feat, cand_mask, bucket_mask, ship_counts, bucket_counts, candidate_ids, target_angles = (
            build_candidate_features(src, state, env_cfg, cache)
        )
        bkt_feat = build_bucket_features(src, state, bucket_counts, bucket_mask, env_cfg)
        self_rows.append(build_self_features(src, state, env_cfg, cache))
        candidate_rows.append(cand_feat)
        candidate_masks.append(cand_mask)
        ship_bucket_masks.append(bucket_mask)
        bucket_feat_rows.append(bkt_feat)
        contexts.append(
            DecisionContext(
                env_index=env_index,
                source_id=src.id,
                candidate_ids=candidate_ids,
                candidate_mask=cand_mask,
                ship_bucket_mask=bucket_mask,
                ship_counts=ship_counts,
                ship_count_buckets=bucket_counts,
                target_angles=target_angles,
            )
        )

    return TurnBatch(
        self_features=np.asarray(self_rows, dtype=np.float32),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32),
        global_features=np.repeat(global_feat[None, :], len(self_rows), axis=0),
        candidate_mask=np.asarray(candidate_masks, dtype=bool),
        ship_bucket_mask=np.asarray(ship_bucket_masks, dtype=bool),
        bucket_features=np.asarray(bucket_feat_rows, dtype=np.float32),
        contexts=contexts,
        state=state,
    )


def build_self_features(
    src: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
    cache: FeatureCache,
) -> np.ndarray:
    my_planets = [planet for planet in state.planets if planet.owner == state.player]
    enemy_planets = [planet for planet in state.planets if planet.owner not in {-1, state.player}]
    neutral_planets = [planet for planet in state.planets if planet.owner == -1]
    surplus = float(source_surplus(src, state))
    reserve = float(source_reserve(src, state))
    center_dx = src.x - BOARD_CENTER[0]
    center_dy = src.y - BOARD_CENTER[1]
    ship_rank = source_ship_rank(src, my_planets)
    production_rank = source_production_rank(src, my_planets)
    incoming_enemy = float(incoming_pressure(src, state, owner_is_enemy=True))
    incoming_friendly = float(incoming_pressure(src, state, owner_is_enemy=False))
    return np.asarray(
        [
            1.0,
            src.x / env_cfg.board_size,
            src.y / env_cfg.board_size,
            src.radius / 5.0,
            min(src.ships, env_cfg.max_ships) / env_cfg.max_ships,
            src.production / env_cfg.max_production,
            1.0 if is_rotating_planet(src) else 0.0,
            surplus / env_cfg.max_ships,
            reserve / env_cfg.max_ships,
            math.hypot(center_dx, center_dy) / env_cfg.board_size,
            ship_rank,
            incoming_enemy / env_cfg.max_ships,
            incoming_friendly / env_cfg.max_ships,
            nearest_planet_distance(src, enemy_planets) / env_cfg.board_size,
            nearest_planet_distance(src, neutral_planets) / env_cfg.board_size,
            cache.planet_centrality.get(src.id, planet_centrality(src)),
            1.0 if src.id in cache.under_attack_ids else 0.0,
            1.0 if is_frontline(src, state) else 0.0,
            friendly_reinforcement_need(src, state) / env_cfg.max_ships,
            count_reachable_sources(src, state, state.player) / env_cfg.max_planets,
            production_rank,
            (incoming_enemy - incoming_friendly) / env_cfg.max_ships,
        ],
        dtype=np.float32,
    )


def build_candidate_features(
    src: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
    cache: FeatureCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[list[int]], list[int], list[float]]:
    """One slot per planet: 0 = no-op, slot planet_id+1 = that planet."""
    slots = target_slot_count(env_cfg)
    buckets = ship_bucket_count(env_cfg)
    features = np.zeros((slots, candidate_feature_dim()), dtype=np.float32)
    candidate_mask = np.zeros((slots,), dtype=bool)
    bucket_mask = np.zeros((slots, buckets), dtype=bool)
    ship_counts = [0] * slots
    bucket_counts = [[0] * buckets for _ in range(slots)]
    candidate_ids = [-1] * slots
    target_angles = [0.0] * slots
    candidate_mask[0] = True
    if buckets > 0:
        bucket_mask[0, 0] = True

    for tgt in state.planets:
        if tgt.id == src.id:
            continue
        idx = planet_slot_for_id(tgt.id)
        if idx >= slots:
            continue

        dx = tgt.x - src.x
        dy = tgt.y - src.y
        if getattr(env_cfg, "use_heuristic_planner", False) and tgt.owner != state.player:
            planned = heuristic_plan_for_target(state, src.id, tgt.id)
        else:
            planned = None
        if planned is None:
            angle = math.atan2(dy, dx)
            heuristic_valid = tgt.owner == state.player or not getattr(env_cfg, "use_heuristic_planner", False)
        else:
            angle, _ships_needed = planned
            heuristic_valid = True

        crosses_sun = shot_crosses_sun(src, angle, tgt)
        ships_for_buckets = ship_bucket_counts(src, tgt, state, env_cfg)
        for bucket_idx, ships in enumerate(ships_for_buckets[:buckets]):
            bucket_counts[idx][bucket_idx] = ships
            bucket_mask[idx, bucket_idx] = (
                heuristic_valid
                and ships > 0
                and not crosses_sun
                and src.ships >= ships
                and is_valid_mission(src, tgt, state, ships)
            )

        dist = distance(src, tgt)
        mission_ships = float(mission_base_ship_count(src, tgt, state))
        travel_ships = max(1, int(default_ship_count(ships_for_buckets) or mission_ships or 1))
        travel_time = estimate_travel_time(
            src,
            tgt,
            travel_ships,
            state.angular_velocity,
        )
        valid_bucket_count = float(sum(1 for flag in bucket_mask[idx] if flag))
        proto_score, proto_eta, capture_at_arrival = proto_move_score(src, tgt, state)
        enemy_produced_eta = max(0.0, capture_at_arrival - float(max(tgt.ships + 1, 1)))
        features[idx] = np.asarray(
            [
                1.0,
                1.0 if tgt.owner == -1 else 0.0,
                1.0 if tgt.owner == state.player else 0.0,
                1.0 if tgt.owner not in {-1, state.player} else 0.0,
                tgt.x / env_cfg.board_size,
                tgt.y / env_cfg.board_size,
                math.sin(angle),
                math.cos(angle),
                dist / env_cfg.board_size,
                min(travel_time / max(1.0, float(env_cfg.episode_steps)), 1.0),
                min(tgt.ships, env_cfg.max_ships) / env_cfg.max_ships,
                tgt.production / env_cfg.max_production,
                1.0 if is_rotating_planet(tgt) else 0.0,
                1.0 if crosses_sun else 0.0,
                min(mission_ships, env_cfg.max_ships) / env_cfg.max_ships,
                incoming_pressure(tgt, state, owner_is_enemy=True) / env_cfg.max_ships,
                incoming_pressure(tgt, state, owner_is_enemy=False) / env_cfg.max_ships,
                1.0 if tgt.id in state.comet_planet_ids else 0.0,
                valid_bucket_count / max(1.0, float(buckets)),
                proto_score / 100.0,
                capture_at_arrival / env_cfg.max_ships,
                enemy_produced_eta / env_cfg.max_ships,
                cache.planet_centrality.get(tgt.id, planet_centrality(tgt)),
                1.0 if tgt.id in cache.under_attack_ids else 0.0,
                target_losing_without_help(tgt, state, cache),
                needs_coop_attack(src, tgt, state),
                max(0.0, (100.0 - dist) / 100.0),
                (tgt.production / env_cfg.max_production) / max(1.0, travel_time),
            ],
            dtype=np.float32,
        )
        ship_counts[idx] = default_ship_count(ships_for_buckets)
        candidate_mask[idx] = True
        candidate_ids[idx] = tgt.id
        target_angles[idx] = angle

    return features, candidate_mask, bucket_mask, ship_counts, bucket_counts, candidate_ids, target_angles


def build_global_features(state: GameState, env_cfg: EnvConfig) -> np.ndarray:
    cache = build_feature_cache(state, env_cfg)
    return build_lb_global_features(state, env_cfg, cache)


def mission_base_ship_count(src: PlanetState, tgt: PlanetState, state: GameState) -> int:
    if tgt.owner == state.player:
        surplus = source_surplus(src, state)
        need = friendly_reinforcement_need(tgt, state)
        if need > 0:
            return min(surplus, max(need, minimum_useful_send(src)))
        if is_forward_staging(src, tgt, state):
            return min(surplus, max(minimum_useful_send(src), int(round(0.5 * surplus))))
        return 0
    return max(tgt.ships + 1, 20)


def exact_ship_count(src: PlanetState, tgt: PlanetState, state: GameState) -> int:
    """Exact mission bucket: ships needed for capture/reinforce, clamped to surplus."""
    surplus = source_surplus(src, state)
    if surplus <= 0:
        return 0
    base = mission_base_ship_count(src, tgt, state)
    if base <= 0:
        return 0
    return max(1, min(surplus, int(base)))


def ship_bucket_counts(
    src: PlanetState,
    tgt: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
) -> list[int]:
    """25%, 50%, 75%, 100% of surplus, plus exact mission count."""
    surplus = source_surplus(src, state)
    buckets = ship_bucket_count(env_cfg)
    if surplus <= 0:
        return [0] * buckets

    raw_counts: list[int] = []
    for fraction in SHIP_BUCKET_FRACTIONS:
        ships = int(round(float(fraction) * surplus))
        raw_counts.append(max(1, min(surplus, ships)) if ships > 0 else 0)

    exact = exact_ship_count(src, tgt, state)
    while len(raw_counts) < EXACT_BUCKET_INDEX:
        raw_counts.append(0)
    if buckets > EXACT_BUCKET_INDEX:
        raw_counts.append(exact)

    counts: list[int] = []
    for ships in raw_counts[:buckets]:
        counts.append(max(0, min(surplus, int(ships))))
    while len(counts) < buckets:
        counts.append(0)
    return counts


def bucket_fraction_label(bucket_idx: int) -> float:
    if 0 <= bucket_idx < len(SHIP_BUCKET_FRACTIONS):
        return float(SHIP_BUCKET_FRACTIONS[bucket_idx])
    return -1.0


def default_ship_count(counts: list[int]) -> int:
    for idx in (1, 2, 4, 0, 3):
        if idx < len(counts) and counts[idx] > 0:
            return counts[idx]
    return 0


def is_valid_mission(src: PlanetState, tgt: PlanetState, state: GameState, ships: int) -> bool:
    if ships <= 0 or ships > source_surplus(src, state):
        return False
    if tgt.owner != state.player:
        return True
    return friendly_reinforcement_need(tgt, state) > 0 or is_forward_staging(src, tgt, state)


def source_surplus(src: PlanetState, state: GameState) -> int:
    return max(0, int(src.ships) - source_reserve(src, state))


def source_reserve(src: PlanetState, state: GameState) -> int:
    enemy_planets = [planet for planet in state.planets if planet.owner not in {-1, state.player}]
    nearest_enemy = min((distance(src, enemy) for enemy in enemy_planets), default=999.0)
    base = 8 + 3 * int(src.production)
    if nearest_enemy < 25.0:
        base += 18
    elif nearest_enemy < 40.0:
        base += 10
    return min(int(src.ships), max(base, int(round(0.20 * src.ships))))


def friendly_reinforcement_need(tgt: PlanetState, state: GameState) -> int:
    enemy_pressure = incoming_pressure(tgt, state, owner_is_enemy=True)
    friendly_help = incoming_pressure(tgt, state, owner_is_enemy=False)
    desired = 12 + 4 * int(tgt.production)
    deficit = enemy_pressure + desired - int(tgt.ships) - friendly_help
    if deficit > 0:
        return int(deficit)
    if is_frontline(tgt, state):
        frontline_desired = 25 + 5 * int(tgt.production)
        return max(0, frontline_desired - int(tgt.ships))
    return 0


def incoming_pressure(tgt: PlanetState, state: GameState, *, owner_is_enemy: bool) -> int:
    total = 0
    for fleet in state.fleets:
        is_enemy = fleet.owner != state.player
        if is_enemy != owner_is_enemy:
            continue
        if fleet_likely_targets_planet(fleet, tgt):
            total += int(fleet.ships)
    return total


def fleet_likely_targets_planet(fleet: Any, tgt: PlanetState) -> bool:
    dx = tgt.x - fleet.x
    dy = tgt.y - fleet.y
    if math.hypot(dx, dy) > 35.0:
        return False
    target_angle = math.atan2(dy, dx)
    return abs(angle_delta(target_angle, fleet.angle)) < 0.20


def is_frontline(planet: PlanetState, state: GameState) -> bool:
    enemy_planets = [other for other in state.planets if other.owner not in {-1, state.player}]
    friendly_planets = [other for other in state.planets if other.owner == state.player and other.id != planet.id]
    nearest_enemy = min((distance(planet, enemy) for enemy in enemy_planets), default=999.0)
    nearest_friend = min((distance(planet, friendly) for friendly in friendly_planets), default=999.0)
    return nearest_enemy < nearest_friend + 12.0


def is_forward_staging(src: PlanetState, tgt: PlanetState, state: GameState) -> bool:
    if src.id == tgt.id or tgt.owner != state.player:
        return False
    enemy_planets = [planet for planet in state.planets if planet.owner not in {-1, state.player}]
    if not enemy_planets:
        return False
    src_enemy_dist = min(distance(src, enemy) for enemy in enemy_planets)
    tgt_enemy_dist = min(distance(tgt, enemy) for enemy in enemy_planets)
    return tgt_enemy_dist + 6.0 < src_enemy_dist


def minimum_useful_send(src: PlanetState) -> int:
    return max(5, min(20, int(src.ships)))


def angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def distance(a: PlanetState, b: PlanetState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def total_ships(planets: list[PlanetState]) -> float:
    return float(sum(planet.ships for planet in planets))


def nearest_planet_distance(src: PlanetState, planets: list[PlanetState]) -> float:
    if not planets:
        return 1.0
    return min(distance(src, planet) for planet in planets)


def source_ship_rank(src: PlanetState, my_planets: list[PlanetState]) -> float:
    if len(my_planets) <= 1:
        return 1.0
    ordered = sorted(my_planets, key=lambda planet: (planet.ships, -planet.id))
    for rank, planet in enumerate(ordered, start=1):
        if planet.id == src.id:
            return rank / float(len(ordered))
    return 0.5


def source_production_rank(src: PlanetState, my_planets: list[PlanetState]) -> float:
    if len(my_planets) <= 1:
        return 1.0
    ordered = sorted(my_planets, key=lambda planet: (planet.production, planet.ships, -planet.id))
    for rank, planet in enumerate(ordered, start=1):
        if planet.id == src.id:
            return rank / float(len(ordered))
    return 0.5


def is_rotating_planet(planet: PlanetState) -> bool:
    dx = planet.x - BOARD_CENTER[0]
    dy = planet.y - BOARD_CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun(src: PlanetState, angle: float, tgt: PlanetState) -> bool:
    start_x = src.x + math.cos(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src.y + math.sin(angle) * (src.radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return point_to_segment_distance(BOARD_CENTER, (start_x, start_y), (tgt.x, tgt.y)) < SUN_RADIUS


def point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    segment_len_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if segment_len_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
        / segment_len_sq
    )
    projection = max(0.0, min(1.0, projection))
    closest_x = start[0] + projection * (end[0] - start[0])
    closest_y = start[1] + projection * (end[1] - start[1])
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def planet_at_slot(slot: int, src: PlanetState, state: GameState) -> PlanetState | None:
    planet_id = planet_id_from_slot(slot)
    if planet_id is None:
        return None
    for planet in state.planets:
        if planet.id == planet_id:
            return None if planet.id == src.id else planet
    return None


def resolve_action_target_id(
    source: PlanetState,
    action_angle: float,
    state: GameState,
    *,
    threshold: float = 0.35,
) -> int | None:
    """Map a replay launch angle to the nearest non-source planet."""
    best_id: int | None = None
    best_delta = float("inf")
    for planet in state.planets:
        if planet.id == source.id:
            continue
        bearing = math.atan2(planet.y - source.y, planet.x - source.x)
        delta = abs(angle_delta(action_angle, bearing))
        if delta < best_delta:
            best_delta = delta
            best_id = planet.id
    if best_id is None or best_delta > threshold:
        return None
    return best_id


def candidate_index_for_target(
    target_planet_id: int,
    candidate_ids: list[int],
    candidate_mask: np.ndarray,
) -> int | None:
    """Return planet slot index for ``target_planet_id``."""
    idx = planet_slot_for_id(target_planet_id)
    if idx >= len(candidate_mask):
        return None
    if candidate_ids[idx] == target_planet_id and candidate_mask[idx]:
        return idx
    if candidate_ids[idx] == target_planet_id:
        return idx
    return None


def bucket_index_for_ships(
    source: PlanetState,
    tgt: PlanetState,
    state: GameState,
    num_ships: int,
    env_cfg: EnvConfig,
) -> int:
    """Map replay ship count to bucket via searchsorted on bucket thresholds."""
    buckets = ship_bucket_counts(source, tgt, state, env_cfg)
    bucket_thresholds = np.asarray(buckets[: ship_bucket_count(env_cfg)], dtype=np.int64)
    valid = [(idx, count) for idx, count in enumerate(bucket_thresholds) if count > 0]
    if not valid:
        return 0
    thresholds = np.asarray([count for _, count in valid], dtype=np.int64)
    pos = int(np.searchsorted(thresholds, num_ships, side="left"))
    pos = min(pos, len(valid) - 1)
    if pos > 0 and abs(num_ships - valid[pos - 1][1]) < abs(num_ships - valid[pos][1]):
        pos -= 1
    return int(valid[pos][0])


def build_bucket_features(
    src: PlanetState,
    state: GameState,
    bucket_counts: list[list[int]],
    bucket_mask: np.ndarray,
    env_cfg: EnvConfig,
) -> np.ndarray:
    """Per-slot, per-bucket features (fraction label + ship count)."""
    slots = target_slot_count(env_cfg)
    buckets = ship_bucket_count(env_cfg)
    bkt_dim = bucket_feature_dim()
    feat = np.zeros((slots, buckets, bkt_dim), dtype=np.float32)
    surplus = max(1.0, float(source_surplus(src, state)))
    for slot_idx in range(slots):
        tgt = planet_at_slot(slot_idx, src, state)
        exact_base = float(exact_ship_count(src, tgt, state)) if tgt is not None else 0.0
        exact_base = max(1.0, exact_base)
        for b_idx in range(buckets):
            ships = float(bucket_counts[slot_idx][b_idx])
            frac = bucket_fraction_label(b_idx)
            feat[slot_idx, b_idx, 0] = min(ships / env_cfg.max_ships, 1.0)
            feat[slot_idx, b_idx, 1] = frac if frac >= 0.0 else min(ships / surplus, 2.0)
            feat[slot_idx, b_idx, 2] = min(ships / exact_base, 3.0) if b_idx == EXACT_BUCKET_INDEX else frac
            feat[slot_idx, b_idx, 3] = 1.0 if bucket_mask[slot_idx, b_idx] else 0.0
    return feat
