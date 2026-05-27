from __future__ import annotations

import importlib.util
import math
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any


Action = list[list[int | float]]
AgentFn = Callable[[Any], Action]


def ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def obs_to_dict(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    out: dict[str, Any] = {}
    for key in (
        "player",
        "planets",
        "fleets",
        "angular_velocity",
        "initial_planets",
        "next_fleet_id",
        "comets",
        "comet_planet_ids",
        "remainingOverageTime",
        "step",
    ):
        if hasattr(obs, key):
            out[key] = getattr(obs, key)
    return out


def clone_value(value: Any) -> Any:
    if isinstance(value, list):
        return [clone_value(x) for x in value]
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    return value


def clone_observation(obs: Any) -> SimpleNamespace:
    return ns(**{k: clone_value(v) for k, v in obs_to_dict(obs).items()})


def clone_state(state: Any) -> SimpleNamespace:
    return ns(
        observation=clone_observation(state.observation),
        action=clone_value(getattr(state, "action", None)),
        status=str(getattr(state, "status", "ACTIVE")),
        reward=clone_value(getattr(state, "reward", 0)),
    )


def noop_agent(_obs: Any) -> Action:
    return []


def deterministic_probe_agent(obs: Any) -> Action:
    payload = obs_to_dict(obs)
    player = int(payload.get("player", 0))
    step = int(payload.get("step", 0))
    planets = list(payload.get("planets") or [])
    mine = [p for p in planets if int(p[1]) == player]
    targets = [
        p
        for p in planets
        if int(p[1]) != player and float(p[2]) >= 0.0 and float(p[3]) >= 0.0
    ]
    moves: Action = []
    if not targets:
        return moves

    for source in sorted(mine, key=lambda p: int(p[0])):
        ships = int(source[5])
        if ships < 8:
            continue
        cadence = 3 + ((int(source[0]) + player) % 4)
        if (step + int(source[0]) + player) % cadence != 0:
            continue
        sx, sy = float(source[2]), float(source[3])
        target = min(
            targets,
            key=lambda t: (
                (float(t[2]) - sx) ** 2 + (float(t[3]) - sy) ** 2,
                -int(t[6]),
                int(t[0]),
            ),
        )
        send = max(1, min(ships // 3, int(target[5]) + 2))
        angle = math.atan2(float(target[3]) - sy, float(target[2]) - sx)
        moves.append([int(source[0]), float(angle), int(send)])
        if len(moves) >= 2:
            break
    return moves


def load_agent_from_file(path: Path, root: Path) -> AgentFn:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module_name = f"direct_agent_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError(f"{path} does not expose callable agent(obs)")
    return agent


def resolve_agent(name: str, root: Path) -> AgentFn:
    if name == "noop":
        return noop_agent
    if name == "probe":
        return deterministic_probe_agent
    if name in {"random", "starter"}:
        from kaggle_environments.envs.orbit_wars.orbit_wars import agents

        base = agents[name]

        def wrapped(obs: Any) -> Action:
            return base(obs_to_dict(obs))

        return wrapped

    path = Path(name)
    if not path.is_absolute():
        path = root / path
    return load_agent_from_file(path, root)


def make_direct_env(
    seed: int | None = None,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    comet_speed: float = 4.0,
) -> SimpleNamespace:
    cfg = ns(
        seed=seed,
        episodeSteps=episode_steps,
        shipSpeed=ship_speed,
        cometSpeed=comet_speed,
        actTimeout=1,
    )
    return ns(configuration=cfg, done=False, info={}, debug=False)


def initial_state(num_agents: int) -> list[SimpleNamespace]:
    state: list[SimpleNamespace] = []
    for i in range(num_agents):
        obs_kwargs = {
            "player": i,
            "planets": [],
            "fleets": [],
            "angular_velocity": 0.0,
            "initial_planets": [],
            "next_fleet_id": 0,
            "comets": [],
            "comet_planet_ids": [],
            "remainingOverageTime": 60,
        }
        if i == 0:
            obs_kwargs["step"] = 0
        obs = ns(**obs_kwargs)
        state.append(ns(observation=obs, action=[], status="ACTIVE", reward=0))
    return state


def done(state: list[SimpleNamespace]) -> bool:
    return any(s.status == "DONE" for s in state)


def agent_observation(state: list[SimpleNamespace], index: int) -> SimpleNamespace:
    obs = clone_observation(state[index].observation)
    shared = state[0].observation
    for key in ("step",):
        if hasattr(shared, key) and not hasattr(obs, key):
            setattr(obs, key, clone_value(getattr(shared, key)))
    return obs


def run_direct(
    agents: list[AgentFn],
    seed: int | None = None,
    episode_steps: int = 500,
    keep_steps: bool = True,
) -> tuple[list[list[SimpleNamespace]], float]:
    from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

    env = make_direct_env(seed=seed, episode_steps=episode_steps)
    state = initial_state(len(agents))

    # Initialization call mirrors Environment.reset().
    state = interpreter(state, env)
    state[0].observation.step = 0
    steps: list[list[SimpleNamespace]] = [[clone_state(s) for s in state]] if keep_steps else []
    step_index = 0

    start = time.perf_counter()
    while not done(state):
        actions: list[Action] = []
        for i, agent in enumerate(agents):
            if state[i].status in {"ACTIVE", "INACTIVE"}:
                actions.append(agent(agent_observation(state, i)))
            else:
                actions.append([])

        action_state = [clone_state(s) for s in state]
        for i, action in enumerate(actions):
            action_state[i].action = action

        env.done = done(action_state)
        state = interpreter(action_state, env)
        step_index += 1
        state[0].observation.step = 0 if env.done else step_index
        env.done = done(state)

        if state[0].observation.step >= episode_steps - 1:
            for s in state:
                if s.status in {"ACTIVE", "INACTIVE"}:
                    s.status = "DONE"
            env.done = True

        if keep_steps:
            steps.append([clone_state(s) for s in state])
        else:
            steps = [[clone_state(s) for s in state]]

    elapsed = time.perf_counter() - start
    return steps, elapsed


def run_direct_from_names(
    agent_names: list[str],
    root: Path,
    seed: int | None = None,
    episode_steps: int = 500,
    keep_steps: bool = True,
) -> tuple[list[list[SimpleNamespace]], float]:
    agents = [resolve_agent(name, root) for name in agent_names]
    return run_direct(agents, seed=seed, episode_steps=episode_steps, keep_steps=keep_steps)
