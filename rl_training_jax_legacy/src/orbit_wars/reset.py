"""Reset Orbit Wars JAX state from reference env."""

from __future__ import annotations

from .convert import observation_to_state
from .reference import episode_seed_from_env, reference_reset
from .state import OrbitWarsState


def reset(
    seed: int,
    *,
    episode_steps: int = 500,
    ship_speed: float = 6.0,
    env_root: str | None = None,
) -> OrbitWarsState:
    env, ref = reference_reset(seed, episode_steps=episode_steps, env_root=env_root)
    episode_seed = episode_seed_from_env(env)
    return observation_to_state(
        ref.observations[0],
        episode_seed=episode_seed,
        ship_speed=ship_speed,
        episode_steps=episode_steps,
        done=ref.done,
        rewards=ref.rewards,
    )
