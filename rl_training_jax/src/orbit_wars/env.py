"""High-level Orbit Wars JAX environment API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .convert import state_to_observation_dict
from .reference import reference_step
from .reset import reset
from .state import OrbitWarsState
from .step import step


@dataclass(slots=True)
class EnvStep:
    state: OrbitWarsState
    observation: dict[str, Any]
    rewards: tuple[float, float]
    done: bool


class OrbitWarsJaxEnv:
    """Single-environment wrapper matching RL training expectations."""

    def __init__(
        self,
        *,
        seed: int = 0,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        env_root: str | None = None,
        learner_player: int = 0,
    ) -> None:
        self.seed = int(seed)
        self.episode_steps = int(episode_steps)
        self.ship_speed = float(ship_speed)
        self.env_root = env_root
        self.learner_player = int(learner_player)
        self.state: OrbitWarsState | None = None
        self._episode = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        else:
            self.seed = self.seed + 9973
        self.state = reset(
            self.seed,
            episode_steps=self.episode_steps,
            ship_speed=self.ship_speed,
            env_root=self.env_root,
        )
        self._episode += 1
        return state_to_observation_dict(self.state, player=self.learner_player)

    def step(self, learner_action: list[list[float | int]], opponent_action: list[list[float | int]]) -> EnvStep:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.learner_player == 0:
            actions = [learner_action, opponent_action]
        else:
            actions = [opponent_action, learner_action]
        self.state = step(self.state, actions)
        obs = state_to_observation_dict(self.state, player=self.learner_player)
        rewards = (float(self.state.rewards[0]), float(self.state.rewards[1]))
        done = bool(self.state.done)
        return EnvStep(state=self.state, observation=obs, rewards=rewards, done=done)


class VectorOrbitWarsEnv:
    """Batched env stepping for throughput benchmarks (JIT core, no comet spawn)."""

    def __init__(self, num_envs: int, *, episode_steps: int = 500) -> None:
        self.num_envs = int(num_envs)
        self.episode_steps = int(episode_steps)
        self.states: OrbitWarsState | None = None

    def reset_batch(self, seeds: list[int]) -> list[dict[str, Any]]:
        assert len(seeds) == self.num_envs
        import jax.numpy as jnp
        from .state import empty_state

        states = [reset(s, episode_steps=self.episode_steps) for s in seeds]
        # stack into batched struct — for benchmark use list stepping if stack fails
        self._state_list = states
        return [state_to_observation_dict(s, player=0) for s in states]

    def step_batch_noop(self) -> None:
        """Advance all envs with empty actions (benchmark helper)."""
        from .step import step

        empty: list[list[float | int]] = []
        self._state_list = [
            step(s, [empty, empty]) for s in self._state_list
        ]
