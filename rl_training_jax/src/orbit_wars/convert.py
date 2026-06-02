"""Convert between Python list observations and padded JAX state."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .constants import (
    FLEET_COLS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_COMET_PLANETS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_PLAYERS,
    PLANET_COLS,
)
from .state import CometGroups, OrbitWarsState, empty_comet_groups, empty_state


def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _pack_planets(rows: list[list[float]], *, pad: np.ndarray) -> tuple[np.ndarray, int]:
    out = pad.copy()
    n = min(len(rows), MAX_PLANETS)
    for i, row in enumerate(rows[:n]):
        out[i, 0] = float(row[0])
        out[i, 1] = float(row[1])
        out[i, 2] = float(row[2])
        out[i, 3] = float(row[3])
        out[i, 4] = float(row[4])
        out[i, 5] = float(row[5])
        out[i, 6] = float(row[6])
        out[i, 7] = 1.0
    return out, n


def _pack_fleets(rows: list[list[float]], *, pad: np.ndarray) -> tuple[np.ndarray, int]:
    out = pad.copy()
    n = min(len(rows), MAX_FLEETS)
    for i, row in enumerate(rows[:n]):
        out[i, 0] = float(row[0])
        out[i, 1] = float(row[1])
        out[i, 2] = float(row[2])
        out[i, 3] = float(row[3])
        out[i, 4] = float(row[4])
        out[i, 5] = float(row[5])
        out[i, 6] = float(row[6])
        out[i, 7] = 1.0
    return out, n


def pack_comets(comets: list[dict[str, Any]]) -> CometGroups:
    active = np.zeros((MAX_COMET_GROUPS,), dtype=np.bool_)
    planet_ids = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)
    path_index = np.full((MAX_COMET_GROUPS,), -1, dtype=np.int32)
    paths = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2), dtype=np.float32)
    path_lengths = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)

    for gi, group in enumerate(comets[:MAX_COMET_GROUPS]):
        active[gi] = True
        path_index[gi] = int(group.get("path_index", -1))
        pids = list(group.get("planet_ids") or [])
        group_paths = list(group.get("paths") or [])
        for pi in range(min(4, len(pids))):
            planet_ids[gi, pi] = int(pids[pi])
            if pi < len(group_paths):
                path = group_paths[pi]
                plen = min(len(path), MAX_COMET_PATH_LEN)
                path_lengths[gi, pi] = plen
                for ti in range(plen):
                    paths[gi, pi, ti, 0] = float(path[ti][0])
                    paths[gi, pi, ti, 1] = float(path[ti][1])

    return CometGroups(
        active=jnp.asarray(active),
        planet_ids=jnp.asarray(planet_ids),
        path_index=jnp.asarray(path_index),
        paths=jnp.asarray(paths),
        path_lengths=jnp.asarray(path_lengths),
    )


def pack_comet_planet_ids(ids: list[int]) -> tuple[jnp.ndarray, int]:
    arr = np.full((MAX_COMET_PLANETS,), -1, dtype=np.int32)
    n = min(len(ids), MAX_COMET_PLANETS)
    for i, pid in enumerate(ids[:n]):
        arr[i] = int(pid)
    return jnp.asarray(arr), n


def observation_to_state(
    obs: Any,
    *,
    episode_seed: int = 0,
    ship_speed: float = 6.0,
    episode_steps: int = 500,
    done: bool = False,
    rewards: tuple[float, float] = (0.0, 0.0),
) -> OrbitWarsState:
    base = empty_state()
    planet_pad = np.zeros((MAX_PLANETS, PLANET_COLS), dtype=np.float32)
    fleet_pad = np.zeros((MAX_FLEETS, FLEET_COLS), dtype=np.float32)

    planets, n_planets = _pack_planets(list(_get(obs, "planets", []) or []), pad=planet_pad)
    initial, _ = _pack_planets(list(_get(obs, "initial_planets", []) or []), pad=planet_pad.copy())
    fleets, n_fleets = _pack_fleets(list(_get(obs, "fleets", []) or []), pad=fleet_pad)
    comets = pack_comets(list(_get(obs, "comets", []) or []))
    comet_ids, n_comet_ids = pack_comet_planet_ids(list(_get(obs, "comet_planet_ids", []) or []))

    return OrbitWarsState(
        planets=jnp.asarray(planets),
        initial_planets=jnp.asarray(initial),
        n_planets=jnp.int32(n_planets),
        fleets=jnp.asarray(fleets),
        n_fleets=jnp.int32(n_fleets),
        comets=comets,
        comet_planet_ids=comet_ids,
        n_comet_planet_ids=jnp.int32(n_comet_ids),
        angular_velocity=jnp.float32(float(_get(obs, "angular_velocity", 0.0))),
        step=jnp.int32(int(_get(obs, "step", 0))),
        next_fleet_id=jnp.int32(int(_get(obs, "next_fleet_id", 0))),
        episode_seed=jnp.int32(int(episode_seed)),
        done=jnp.bool_(done),
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
        ship_speed=jnp.float32(float(ship_speed)),
        episode_steps=jnp.int32(int(episode_steps)),
    )


def _planet_rows(state: OrbitWarsState) -> list[list[float]]:
    rows: list[list[float]] = []
    planets = np.asarray(state.planets)
    for i in range(int(state.n_planets)):
        if planets[i, 7] <= 0.0:
            continue
        rows.append(
            [
                float(planets[i, 0]),
                float(planets[i, 1]),
                float(planets[i, 2]),
                float(planets[i, 3]),
                float(planets[i, 4]),
                float(planets[i, 5]),
                float(planets[i, 6]),
            ]
        )
    return rows


def _fleet_rows(state: OrbitWarsState) -> list[list[float]]:
    rows: list[list[float]] = []
    fleets = np.asarray(state.fleets)
    for i in range(int(state.n_fleets)):
        if fleets[i, 7] <= 0.0:
            continue
        rows.append(
            [
                float(fleets[i, 0]),
                float(fleets[i, 1]),
                float(fleets[i, 2]),
                float(fleets[i, 3]),
                float(fleets[i, 4]),
                float(fleets[i, 5]),
                float(fleets[i, 6]),
            ]
        )
    return rows


def _unpack_comets(comets: CometGroups) -> list[dict[str, Any]]:
    active = np.asarray(comets.active)
    planet_ids = np.asarray(comets.planet_ids)
    path_index = np.asarray(comets.path_index)
    paths = np.asarray(comets.paths)
    path_lengths = np.asarray(comets.path_lengths)
    out: list[dict[str, Any]] = []
    for gi in range(MAX_COMET_GROUPS):
        if not active[gi]:
            continue
        group_paths: list[list[list[float]]] = []
        pids: list[int] = []
        for pi in range(4):
            pid = int(planet_ids[gi, pi])
            if pid < 0:
                continue
            pids.append(pid)
            plen = int(path_lengths[gi, pi])
            group_paths.append(
                [[float(paths[gi, pi, ti, 0]), float(paths[gi, pi, ti, 1])] for ti in range(plen)]
            )
        out.append({"planet_ids": pids, "paths": group_paths, "path_index": int(path_index[gi])})
    return out


def state_to_observation_dict(state: OrbitWarsState, *, player: int = 0) -> dict[str, Any]:
    comet_ids = [int(x) for x in np.asarray(state.comet_planet_ids)[: int(state.n_comet_planet_ids)] if int(x) >= 0]
    return {
        "step": int(state.step),
        "player": int(player),
        "planets": _planet_rows(state),
        "initial_planets": _planet_rows(
            OrbitWarsState(
                planets=state.initial_planets,
                initial_planets=state.initial_planets,
                n_planets=state.n_planets,
                fleets=state.fleets,
                n_fleets=state.n_fleets,
                comets=state.comets,
                comet_planet_ids=state.comet_planet_ids,
                n_comet_planet_ids=state.n_comet_planet_ids,
                angular_velocity=state.angular_velocity,
                step=state.step,
                next_fleet_id=state.next_fleet_id,
                episode_seed=state.episode_seed,
                done=state.done,
                rewards=state.rewards,
                ship_speed=state.ship_speed,
                episode_steps=state.episode_steps,
            )
        ),
        "fleets": _fleet_rows(state),
        "angular_velocity": float(state.angular_velocity),
        "next_fleet_id": int(state.next_fleet_id),
        "comets": _unpack_comets(state.comets),
        "comet_planet_ids": comet_ids,
    }


def states_equal(a: OrbitWarsState, b: OrbitWarsState, *, atol: float = 1e-4) -> bool:
    """Compare simulation-relevant fields (ignores padded inactive slots)."""
    if int(a.n_planets) != int(b.n_planets) or int(a.n_fleets) != int(b.n_fleets):
        return False
    if int(a.step) != int(b.step) or bool(a.done) != bool(b.done):
        return False
    ap = np.asarray(a.planets)[: int(a.n_planets), :7]
    bp = np.asarray(b.planets)[: int(b.n_planets), :7]
    af = np.asarray(a.fleets)[: int(a.n_fleets), :7]
    bf = np.asarray(b.fleets)[: int(b.n_fleets), :7]
    if not np.allclose(ap, bp, atol=atol, rtol=0.0):
        return False
    if not np.allclose(af, bf, atol=atol, rtol=0.0):
        return False
    return True
