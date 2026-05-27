from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.constants import CENTER, SUN_RADIUS
from src.game import Fleet, GameState, Planet
from src.geometry import (
    angle_between,
    distance_xy,
    estimate_intercept,
    fleet_ray_closest_to_point,
    segment_clear_of_circles,
)

NEUTRAL = -1
GAME_TURNS = 500

DEFAULT_CONFIG: dict[str, Any] = {
    # Reserves and launches
    "min_defense": 12,
    "launch_fraction": 0.65,
    # Scoring weights
    "production_weight": 32.0,
    "distance_weight": 0.45,
    "ships_weight": 1.0,
    "enemy_bonus": 8.0,
    "comet_bonus": 3.0,
    "payoff_horizon": 120.0,
    "pay_rate": 0.12,
    # Capture safety
    "safety_margin": 2,
    "patient_capture_plus": 1,
    # Path safety
    "avoid_sun_margin": 0.4,
    "path_circle_margin": 0.6,
    # Reachability filters
    "expansion_max_manhattan": 88,
    "travel_time_max": 60.0,
    "payoff_min_turns": 20,
    # Smooth buffer ramp
    "buffer_zero_turns": 18,
    "buffer_early_floor": 0,
    "buffer_base_after_early": 0,
    "buffer_ramp_turns": 30,
    "buffer_linear_per_turn": 0.08,
    "buffer_production_mult": 1.0,
    # Threats and defense
    "threat_horizon_turns": 90.0,
    "threat_hit_slop": 2.0,
    "min_threat_ships": 3,
    "defense_reinforce_margin": 3,
    # Comet handling without lifetime info
    "comet_travel_time_max": 25.0,
    # Score-based adaptation (military lead vs strongest opponent)
    "adaptive_enabled": True,
    "adaptive_behind_ratio": -0.04,
    "adaptive_ahead_ratio": 0.04,
    "adaptive_behind_buffer_mult": 1.12,
    "adaptive_ahead_buffer_mult": 0.9,
    "adaptive_behind_travel_scale": 0.82,
    "adaptive_ahead_travel_scale": 1.1,
    "adaptive_behind_payoff_min_delta": -4.0,
    "adaptive_ahead_payoff_min_delta": 5.0,
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


# ---------- Buffer (smooth ramp) ----------

def per_planet_ship_buffer(step: int, production: int, cfg: dict[str, Any]) -> int:
    """Per-planet reserve. Zero in the opening; smooth ramp afterwards, plus production term."""
    z = int(cfg["buffer_zero_turns"])
    if step <= z:
        raw = int(cfg["buffer_early_floor"])
    else:
        after = step - z
        ramp = max(1, int(cfg["buffer_ramp_turns"]))
        t = min(1.0, after / float(ramp))
        base = (int(cfg["min_defense"]) + int(cfg["buffer_base_after_early"])) * t
        linear = float(cfg["buffer_linear_per_turn"]) * after
        prod_term = float(cfg["buffer_production_mult"]) * max(0, int(production))
        raw = int(base + linear + prod_term)
    mult = float(cfg.get("adaptive_buffer_mult", 1.0))
    return max(0, int(round(raw * mult)))


def total_military_power(state: GameState, player: int) -> int:
    s = 0
    for p in state.planets:
        if p.owner == player:
            s += int(p.ships)
    for f in state.fleets:
        if f.owner == player:
            s += int(f.ships)
    return s


def effective_cfg(state: GameState, base: dict[str, Any]) -> dict[str, Any]:
    """Per-turn cfg copy with adaptive travel cap, buffer scale, and late-game cutoff."""
    cfg = dict(base)
    cfg.setdefault("adaptive_buffer_mult", 1.0)
    if not bool(cfg.get("adaptive_enabled", True)):
        return cfg

    me = int(state.player)
    players: set[int] = set()
    for p in state.planets:
        if p.owner >= 0:
            players.add(int(p.owner))
    for f in state.fleets:
        if f.owner >= 0:
            players.add(int(f.owner))

    my_pow = total_military_power(state, me)
    best_opp = 0
    for pid in players:
        if pid == me:
            continue
        best_opp = max(best_opp, total_military_power(state, pid))

    denom = max(1, my_pow + best_opp)
    ratio = (my_pow - best_opp) / float(denom)
    behind_thr = float(cfg.get("adaptive_behind_ratio", -0.04))
    ahead_thr = float(cfg.get("adaptive_ahead_ratio", 0.04))

    tmax = float(cfg["travel_time_max"])
    pmin = float(cfg["payoff_min_turns"])
    buf_m = 1.0

    if ratio < behind_thr:
        buf_m = float(cfg.get("adaptive_behind_buffer_mult", 1.12))
        tmax *= float(cfg.get("adaptive_behind_travel_scale", 0.82))
        pmin += float(cfg.get("adaptive_behind_payoff_min_delta", -4.0))
    elif ratio > ahead_thr:
        buf_m = float(cfg.get("adaptive_ahead_buffer_mult", 0.9))
        tmax *= float(cfg.get("adaptive_ahead_travel_scale", 1.1))
        pmin += float(cfg.get("adaptive_ahead_payoff_min_delta", 5.0))

    cfg["adaptive_buffer_mult"] = buf_m
    cfg["travel_time_max"] = max(12.0, min(95.0, tmax))
    cfg["payoff_min_turns"] = max(5.0, min(85.0, pmin))
    return cfg


# ---------- In-flight fleet bookkeeping ----------

def fleet_target_planet(
    fleet: Fleet,
    planets: list[Planet],
    slop: float,
) -> tuple[int, float] | None:
    """Find which planet a fleet is heading to (smallest eta with d_close <= radius+slop)."""
    best: tuple[float, int] | None = None
    for p in planets:
        t_close, d_close = fleet_ray_closest_to_point(fleet, p.x, p.y)
        if t_close <= 0:
            continue
        if d_close <= p.radius + slop:
            if best is None or t_close < best[0]:
                best = (t_close, p.id)
    if best is None:
        return None
    return best[1], best[0]


def build_arrivals_by_planet(
    state: GameState, slop: float
) -> dict[int, list[tuple[float, int, int]]]:
    """planet_id -> sorted list of (eta, owner, ships) arrivals from in-flight fleets."""
    arr: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for f in state.fleets:
        found = fleet_target_planet(f, state.planets, slop)
        if found is None:
            continue
        pid, eta = found
        arr[pid].append((eta, int(f.owner), int(f.ships)))
    for pid in arr:
        arr[pid].sort(key=lambda x: x[0])
    return arr


def simulate_planet_at(
    planet: Planet,
    upto_time: float,
    arrivals: list[tuple[float, int, int]],
) -> tuple[int, int]:
    """(owner, ships) on `planet` after applying in-flight arrivals up to `upto_time`.
    Production accrues only for non-neutral owners."""
    owner = int(planet.owner)
    ships = int(planet.ships)
    last_t = 0.0
    for eta, f_owner, f_ships in arrivals:
        if eta > upto_time:
            break
        if owner != NEUTRAL:
            ships += int(planet.production * (eta - last_t))
        last_t = eta
        if f_owner == owner:
            ships += int(f_ships)
        else:
            if f_ships > ships:
                owner = int(f_owner)
                ships = int(f_ships) - ships
            else:
                ships -= int(f_ships)
    if owner != NEUTRAL:
        ships += int(planet.production * (upto_time - last_t))
    return owner, max(0, ships)


# ---------- Threats ----------

def aggregate_threats_by_planet(
    state: GameState, cfg: dict[str, Any]
) -> dict[int, list[tuple[float, Fleet]]]:
    """Owned planet id -> [(eta, enemy_fleet), ...] sorted by eta."""
    out: dict[int, list[tuple[float, Fleet]]] = defaultdict(list)
    slop = float(cfg["threat_hit_slop"])
    min_ships = int(cfg["min_threat_ships"])
    horizon = float(cfg["threat_horizon_turns"])
    for f in state.fleets:
        if f.owner == state.player:
            continue
        if int(f.ships) < min_ships:
            continue
        for p in state.planets:
            if p.owner != state.player:
                continue
            t_close, d_close = fleet_ray_closest_to_point(f, p.x, p.y)
            if t_close <= 0 or t_close > horizon:
                continue
            if d_close > p.radius + slop:
                continue
            out[p.id].append((t_close, f))
    for pid in out:
        out[pid].sort(key=lambda x: x[0])
    return out


def planet_deficit(
    threatened: Planet,
    our_player: int,
    arrivals: list[tuple[float, int, int]],
) -> tuple[int, float]:
    """If threatened planet ever flips away from us, return (deficit_ships, eta_of_loss).
    Deficit is how many extra ships we'd need on arrival to prevent the flip."""
    owner = int(threatened.owner)
    ships = int(threatened.ships)
    last_t = 0.0
    for eta, f_owner, f_ships in arrivals:
        if owner != NEUTRAL:
            ships += int(threatened.production * (eta - last_t))
        last_t = eta
        if f_owner == owner:
            ships += int(f_ships)
        else:
            if f_ships > ships:
                deficit = int(f_ships) - ships + 1
                if owner == our_player:
                    return deficit, eta
                owner = int(f_owner)
                ships = int(f_ships) - ships
            else:
                ships -= int(f_ships)
    return 0, 0.0


# ---------- Path safety ----------

def obstacles_for_path(
    state: GameState, source_id: int, target_id: int, cfg: dict[str, Any]
) -> list[tuple[tuple[float, float], float]]:
    """Sun and non-(source/target) enemy planets as collision circles."""
    margin = float(cfg["path_circle_margin"])
    sun_margin = float(cfg["avoid_sun_margin"])
    obs: list[tuple[tuple[float, float], float]] = [
        (CENTER, SUN_RADIUS + sun_margin)
    ]
    for p in state.planets:
        if p.id == source_id or p.id == target_id:
            continue
        if p.owner == state.player or p.owner == NEUTRAL:
            continue
        obs.append(((p.x, p.y), p.radius + margin))
    return obs


# ---------- Source survival ----------

def source_survives_after_launch(
    source: Planet,
    take: int,
    threats_at_source: list[tuple[float, Fleet]],
    cfg: dict[str, Any],
) -> bool:
    """After we send `take` ships, does the source still survive its known threats?"""
    if not threats_at_source:
        return True
    remaining = int(source.ships) - int(take)
    if remaining < 0:
        return False
    last_t = 0.0
    margin = int(cfg["defense_reinforce_margin"])
    for eta, f in threats_at_source:
        remaining += int(source.production * (eta - last_t))
        last_t = eta
        remaining -= int(f.ships)
        if remaining < margin:
            return False
    return True


# ---------- Capture sizing ----------

def need_ships_for_capture(
    state: GameState,
    target: Planet,
    travel_time: float,
    arrivals_at_target: list[tuple[float, int, int]],
    patient: bool,
    cfg: dict[str, Any],
) -> tuple[int, int] | None:
    """Returns (owner_at_arrival, ships_needed) or None if already ours when we arrive."""
    owner_at, ships_at = simulate_planet_at(target, travel_time, arrivals_at_target)
    if owner_at == state.player:
        return None
    plus = int(cfg["patient_capture_plus"]) if patient else 1
    margin = int(cfg["safety_margin"])
    return owner_at, ships_at + plus + margin


def find_capture_launch(
    state: GameState,
    source: Planet,
    target: Planet,
    avail: int,
    cfg: dict[str, Any],
    arrivals_at_target: list[tuple[float, int, int]],
    obstacles: list[tuple[tuple[float, float], float]],
    patient: bool,
) -> tuple[float, int, float, tuple[float, float]] | None:
    """Pick the smallest fleet from `source` that captures `target` after flight.
    Single-shot: estimate intercept at `avail`, compute need, refine intercept at `need`.
    Returns (angle, send, travel_time, pred_xy) or None."""
    if avail < 1:
        return None
    travel_max = float(cfg["travel_time_max"])

    angle0, tt0, _d, _spd, pred0 = estimate_intercept(
        source, target, avail, state.angular_velocity
    )
    if tt0 > travel_max:
        return None
    if not segment_clear_of_circles((source.x, source.y), pred0, obstacles):
        return None
    info = need_ships_for_capture(state, target, tt0, arrivals_at_target, patient, cfg)
    if info is None:
        return None
    _owner0, need = info
    if avail < need:
        return None

    send = int(need)
    angle, tt, _d, _spd, pred_xy = estimate_intercept(
        source, target, send, state.angular_velocity
    )
    if tt > travel_max:
        return None
    if not segment_clear_of_circles((source.x, source.y), pred_xy, obstacles):
        return None
    info2 = need_ships_for_capture(state, target, tt, arrivals_at_target, patient, cfg)
    if info2 is None:
        return None
    _owner1, need2 = info2
    if need2 > send:
        send = min(int(avail), int(need2))
        if send > avail:
            return None
        angle, tt, _d, _spd, pred_xy = estimate_intercept(
            source, target, send, state.angular_velocity
        )
        if tt > travel_max:
            return None
        if not segment_clear_of_circles((source.x, source.y), pred_xy, obstacles):
            return None
    return angle, send, tt, pred_xy


# ---------- Score ----------

def composite_score(
    state: GameState,
    target: Planet,
    send_ships: int,
    travel_time: float,
    dist: float,
    cfg: dict[str, Any],
) -> float:
    remaining = max(1.0, GAME_TURNS - state.step - travel_time)
    payoff_h = float(cfg["payoff_horizon"])
    payoff = float(target.production) * min(remaining, payoff_h) * float(cfg["pay_rate"])
    score = payoff
    score -= float(send_ships) * float(cfg["ships_weight"])
    score -= float(dist) * float(cfg["distance_weight"])
    if target.owner not in (NEUTRAL, state.player):
        score += float(cfg["enemy_bonus"])
    if target.id in state.comet_planet_ids:
        score += float(cfg["comet_bonus"])
    return score


def expansion_payoff_roi(
    state: GameState,
    target: Planet,
    send_ships: int,
    travel_time: float,
    cfg: dict[str, Any],
) -> float:
    """Production-weighted payoff per ship sent (Smart Aggressor style ROI hint)."""
    remaining = max(1.0, GAME_TURNS - state.step - travel_time)
    payoff_h = float(cfg["payoff_horizon"])
    payoff = float(target.production) * min(remaining, payoff_h) * float(cfg["pay_rate"])
    if target.id in state.comet_planet_ids:
        payoff += float(cfg.get("comet_bonus", 0.0)) * 0.5
    if target.owner not in (NEUTRAL, state.player):
        payoff += float(cfg.get("enemy_bonus", 0.0)) * 0.15
    return payoff / max(1.0, float(send_ships))


# ---------- Expansion ----------

def candidate_targets(
    state: GameState, my_planets: list[Planet], cfg: dict[str, Any]
) -> list[Planet]:
    max_m = float(cfg["expansion_max_manhattan"])
    out: list[Planet] = []
    for t in state.planets:
        if t.owner == state.player:
            continue
        # Manhattan gate to nearest owned (coarse early filter)
        nearest_m = min(abs(t.x - p.x) + abs(t.y - p.y) for p in my_planets)
        if nearest_m > max_m:
            continue
        out.append(t)
    return out


def try_expansion_for_target(
    state: GameState,
    target: Planet,
    my_planets: list[Planet],
    cfg: dict[str, Any],
    committed: defaultdict[int, int],
    arrivals: dict[int, list[tuple[float, int, int]]],
    threats_by_planet: dict[int, list[tuple[float, Fleet]]],
) -> tuple[int, float, int, float, float] | None:
    """Returns (source_id, angle, send, travel_time, score) or None."""
    arrivals_at_target = arrivals.get(target.id, [])
    is_enemy = target.owner not in (NEUTRAL, state.player)
    patient = is_enemy
    is_comet = target.id in state.comet_planet_ids
    comet_tt_max = float(cfg["comet_travel_time_max"])
    payoff_min = float(cfg["payoff_min_turns"])

    best: tuple[float, int, float, int, float] | None = None
    for source in sorted(
        my_planets,
        key=lambda s: distance_xy((s.x, s.y), (target.x, target.y)),
    ):
        if source.id == target.id:
            continue
        buf = per_planet_ship_buffer(state.step, source.production, cfg)
        avail = int(source.ships) - buf - committed[source.id]
        if avail < 1:
            continue

        obstacles = obstacles_for_path(state, source.id, target.id, cfg)
        got = find_capture_launch(
            state, source, target, avail, cfg, arrivals_at_target, obstacles, patient
        )
        if got is None:
            continue
        angle, send_ships, travel_time, pred_xy = got

        if is_comet and travel_time > comet_tt_max:
            continue
        if state.step + travel_time > GAME_TURNS - payoff_min:
            continue

        threats_here = threats_by_planet.get(source.id, [])
        if not source_survives_after_launch(source, send_ships, threats_here, cfg):
            continue

        dist = distance_xy((source.x, source.y), pred_xy)
        score = composite_score(state, target, send_ships, travel_time, dist, cfg)
        if score <= 0.0:
            continue

        if best is None or score > best[0]:
            best = (score, source.id, angle, send_ships, travel_time)

    if best is None:
        return None
    score, source_id, angle, send_ships, travel_time = best
    return source_id, angle, send_ships, travel_time, score


def choose_expansion_moves(
    state: GameState,
    cfg: dict[str, Any],
    committed: defaultdict[int, int],
    arrivals: dict[int, list[tuple[float, int, int]]],
    threats_by_planet: dict[int, list[tuple[float, Fleet]]],
) -> list[list[int | float]]:
    """Greedy expansion: each step pick the feasible target with best payoff ROI."""
    moves: list[list[int | float]] = []
    my_planets = [p for p in state.planets if p.owner == state.player]
    if not my_planets:
        return moves
    targets = candidate_targets(state, my_planets, cfg)
    if not targets:
        return moves

    taken: set[int] = set()
    while True:
        best_key: tuple[float, float, float] | None = None
        best_pick: tuple[int, float, int, int] | None = None
        for target in targets:
            if target.id in taken:
                continue
            got = try_expansion_for_target(
                state, target, my_planets, cfg, committed, arrivals, threats_by_planet
            )
            if got is None:
                continue
            source_id, angle, send_ships, travel_time, score = got
            roi = expansion_payoff_roi(
                state, target, send_ships, travel_time, cfg
            )
            key = (roi, score, -travel_time)
            if best_key is None or key > best_key:
                best_key = key
                best_pick = (int(target.id), int(source_id), float(angle), int(send_ships))

        if best_pick is None:
            break
        tid, sid, ang, send = best_pick
        moves.append([sid, ang, send])
        committed[sid] += send
        taken.add(tid)

    return moves


# ---------- Defense ----------

def choose_defense_moves(
    state: GameState,
    cfg: dict[str, Any],
    committed: defaultdict[int, int],
    arrivals: dict[int, list[tuple[float, int, int]]],
    threats_by_planet: dict[int, list[tuple[float, Fleet]]],
) -> list[list[int | float]]:
    moves: list[list[int | float]] = []
    my_planets = [p for p in state.planets if p.owner == state.player]
    if not my_planets:
        return moves

    planet_by_id = {p.id: p for p in state.planets}
    margin = int(cfg["defense_reinforce_margin"])
    payoff_min = float(cfg["payoff_min_turns"])

    planet_threats = sorted(
        threats_by_planet.items(),
        key=lambda kv: kv[1][0][0] if kv[1] else 1e9,
    )

    for pid, threats in planet_threats:
        if not threats:
            continue
        threatened = planet_by_id.get(pid)
        if threatened is None or threatened.owner != state.player:
            continue

        arrivals_here = arrivals.get(pid, [])
        deficit, eta_of_loss = planet_deficit(threatened, state.player, arrivals_here)
        if deficit <= 0:
            continue

        earliest_threat_eta = threats[0][0]
        remaining_deficit = deficit + margin

        for defender in sorted(
            my_planets,
            key=lambda d: distance_xy((d.x, d.y), (threatened.x, threatened.y)),
        ):
            if defender.id == threatened.id:
                continue
            if remaining_deficit <= 0:
                break
            buf = per_planet_ship_buffer(state.step, defender.production, cfg)
            avail = int(defender.ships) - buf - committed[defender.id]
            if avail < 1:
                continue

            send_try = min(avail, remaining_deficit)
            angle, tt, _d, _spd, pred_xy = estimate_intercept(
                defender, threatened, send_try, state.angular_velocity
            )
            obstacles = obstacles_for_path(state, defender.id, threatened.id, cfg)
            if not segment_clear_of_circles((defender.x, defender.y), pred_xy, obstacles):
                continue
            if tt >= earliest_threat_eta:
                a2, t2, _d2, _s2, pred2 = estimate_intercept(
                    defender, threatened, avail, state.angular_velocity
                )
                if not segment_clear_of_circles((defender.x, defender.y), pred2, obstacles):
                    continue
                if t2 >= earliest_threat_eta:
                    continue
                send_try = avail
                angle, tt, pred_xy = a2, t2, pred2
            if state.step + tt > GAME_TURNS - payoff_min:
                continue

            threats_at_def = threats_by_planet.get(defender.id, [])
            if not source_survives_after_launch(defender, send_try, threats_at_def, cfg):
                continue

            send_final = send_try
            angle_final = float(angle_between((defender.x, defender.y), pred_xy))
            moves.append([int(defender.id), angle_final, int(send_final)])
            committed[defender.id] += send_final
            remaining_deficit -= send_final

    return moves


# ---------- Top level ----------

def decide_moves(state: GameState) -> list[list[int | float]]:
    base = load_config()
    cfg = effective_cfg(state, base)
    arrivals = build_arrivals_by_planet(state, float(cfg["threat_hit_slop"]))
    threats_by_planet = aggregate_threats_by_planet(state, cfg)
    committed: defaultdict[int, int] = defaultdict(int)
    out: list[list[int | float]] = []
    out.extend(choose_defense_moves(state, cfg, committed, arrivals, threats_by_planet))
    out.extend(choose_expansion_moves(state, cfg, committed, arrivals, threats_by_planet))
    return out
