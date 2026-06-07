"""Reference Kaggle env bridge for reset and parity validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


def load_orbit_wars_module() -> ModuleType:
    """Import official orbit_wars.py (installed package or local checkout)."""
    try:
        from kaggle_environments.envs.orbit_wars import orbit_wars as ref

        return ref
    except ImportError:
        pass

    candidates = [
        Path("/media/yahor/ADATA SE880/datasets/kaggle-environments-master"),
        Path(__file__).resolve().parents[3] / "analysis" / "fast_kaggle_env",
    ]
    for root in candidates:
        module_path = root / "kaggle_environments" / "envs" / "orbit_wars" / "orbit_wars.py"
        if module_path.exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from kaggle_environments.envs.orbit_wars import orbit_wars as ref

            return ref

    raise ImportError(
        "Could not import kaggle_environments.envs.orbit_wars.orbit_wars. "
        "On Kaggle this should be preinstalled; locally install kaggle-environments "
        "or set analysis/fast_kaggle_env."
    )


@dataclass(slots=True)
class ReferenceStep:
    observations: list[Any]
    rewards: tuple[float, float]
    done: bool


def add_env_root(env_root: str | Path) -> Path:
    path = Path(env_root)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def default_env_root() -> Path:
    repo = Path(__file__).resolve().parents[2]
    fast = repo / "analysis" / "fast_kaggle_env"
    official = Path("/media/yahor/ADATA SE880/datasets/kaggle-environments-master")
    if fast.exists():
        return fast
    if official.exists():
        return official
    return fast


def make_reference_env(
    *,
    seed: int,
    episode_steps: int = 500,
    env_root: str | Path | None = None,
) -> Any:
    add_env_root(env_root or default_env_root())
    from kaggle_environments import make

    configuration = {"episodeSteps": int(episode_steps), "seed": int(seed), "randomSeed": int(seed)}
    env = make("orbit_wars", configuration=configuration, debug=False)
    env.reset(num_agents=2)
    return env


def extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def reference_reset(seed: int, *, episode_steps: int = 500, env_root: str | Path | None = None) -> tuple[Any, ReferenceStep]:
    env = make_reference_env(seed=seed, episode_steps=episode_steps, env_root=env_root)
    states = env.step([[], []])
    obs = [extract_observation(states[i]) for i in range(2)]
    rewards = (extract_reward(states[0]), extract_reward(states[1]))
    done = extract_status(states[0]) != "ACTIVE"
    return env, ReferenceStep(observations=obs, rewards=rewards, done=done)


def reference_step(env: Any, actions: list[list[list[float | int]]]) -> ReferenceStep:
    states = env.step(actions)
    obs = [extract_observation(states[i]) for i in range(2)]
    rewards = (extract_reward(states[0]), extract_reward(states[1]))
    done = extract_status(states[0]) != "ACTIVE"
    return ReferenceStep(observations=obs, rewards=rewards, done=done)


def episode_seed_from_env(env: Any) -> int:
    info = getattr(env, "info", None) or {}
    seed = info.get("seed")
    if seed is not None:
        return int(seed)
    return 0
