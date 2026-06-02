"""Padded Orbit Wars game state for JAX simulation."""

from __future__ import annotations

from flax import struct
import jax.numpy as jnp

from .constants import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SHIP_SPEED,
    FLEET_COLS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_COMET_PLANETS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_PLAYERS,
    PLANET_COLS,
)


@struct.dataclass
class CometGroups:
    active: jnp.ndarray  # (MAX_COMET_GROUPS,) bool
    planet_ids: jnp.ndarray  # (MAX_COMET_GROUPS, 4) int32
    path_index: jnp.ndarray  # (MAX_COMET_GROUPS,) int32
    paths: jnp.ndarray  # (MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2) float32
    path_lengths: jnp.ndarray  # (MAX_COMET_GROUPS, 4) int32


@struct.dataclass
class OrbitWarsState:
    planets: jnp.ndarray  # (MAX_PLANETS, PLANET_COLS)
    initial_planets: jnp.ndarray  # (MAX_PLANETS, PLANET_COLS)
    n_planets: jnp.int32
    fleets: jnp.ndarray  # (MAX_FLEETS, FLEET_COLS)
    n_fleets: jnp.int32
    comets: CometGroups
    comet_planet_ids: jnp.ndarray  # (MAX_COMET_PLANETS,) int32, -1 pad
    n_comet_planet_ids: jnp.int32
    angular_velocity: jnp.float32
    step: jnp.int32
    next_fleet_id: jnp.int32
    episode_seed: jnp.int32
    done: jnp.bool_
    rewards: jnp.ndarray  # (NUM_PLAYERS,) float32
    ship_speed: jnp.float32
    episode_steps: jnp.int32


def empty_comet_groups() -> CometGroups:
    return CometGroups(
        active=jnp.zeros((MAX_COMET_GROUPS,), dtype=jnp.bool_),
        planet_ids=jnp.full((MAX_COMET_GROUPS, 4), -1, dtype=jnp.int32),
        path_index=jnp.full((MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        paths=jnp.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH_LEN, 2), dtype=jnp.float32),
        path_lengths=jnp.zeros((MAX_COMET_GROUPS, 4), dtype=jnp.int32),
    )


def empty_state() -> OrbitWarsState:
    return OrbitWarsState(
        planets=jnp.zeros((MAX_PLANETS, PLANET_COLS), dtype=jnp.float32),
        initial_planets=jnp.zeros((MAX_PLANETS, PLANET_COLS), dtype=jnp.float32),
        n_planets=jnp.int32(0),
        fleets=jnp.zeros((MAX_FLEETS, FLEET_COLS), dtype=jnp.float32),
        n_fleets=jnp.int32(0),
        comets=empty_comet_groups(),
        comet_planet_ids=jnp.full((MAX_COMET_PLANETS,), -1, dtype=jnp.int32),
        n_comet_planet_ids=jnp.int32(0),
        angular_velocity=jnp.float32(0.0),
        step=jnp.int32(0),
        next_fleet_id=jnp.int32(0),
        episode_seed=jnp.int32(0),
        done=jnp.bool_(False),
        rewards=jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
        ship_speed=jnp.float32(DEFAULT_SHIP_SPEED),
        episode_steps=jnp.int32(DEFAULT_EPISODE_STEPS),
    )
