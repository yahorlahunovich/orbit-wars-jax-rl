"""Tests for JAX geometry helpers."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from orbit_wars.geometry import fleet_speed, point_to_segment_distance, swept_pair_hit


def test_point_to_segment_distance():
    d = point_to_segment_distance(
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(3.0),
        jnp.array(4.0),
    )
    assert float(d) == pytest.approx(0.0)


def test_swept_pair_hit_crossing():
    hit = swept_pair_hit(
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(10.0),
        jnp.array(0.0),
        jnp.array(5.0),
        jnp.array(-1.0),
        jnp.array(5.0),
        jnp.array(1.0),
        jnp.array(1.0),
    )
    assert bool(hit)


def test_fleet_speed_bounds():
    slow = float(fleet_speed(jnp.array(1.0), jnp.array(6.0)))
    fast = float(fleet_speed(jnp.array(1000.0), jnp.array(6.0)))
    assert slow == pytest.approx(1.0)
    assert fast == pytest.approx(6.0)


def test_swept_pair_matches_python_reference():
    try:
        from orbit_wars.reference import load_orbit_wars_module

        ref = load_orbit_wars_module()
    except ImportError:
        pytest.skip("kaggle_environments not available")
    cases = [
        ((0.0, 0.0, 10.0, 0.0, 5.0, -1.0, 5.0, 1.0, 1.0), True),
        ((0.0, 0.0, 10.0, 0.0, 5.0, 5.0, 5.0, 6.0, 1.0), False),
    ]
    for args, expected in cases:
        py = ref.swept_pair_hit(args[:2], args[2:4], args[4:6], args[6:8], args[8])
        jax = bool(
            swept_pair_hit(
                jnp.array(args[0]),
                jnp.array(args[1]),
                jnp.array(args[2]),
                jnp.array(args[3]),
                jnp.array(args[4]),
                jnp.array(args[5]),
                jnp.array(args[6]),
                jnp.array(args[7]),
                jnp.array(args[8]),
            )
        )
        assert jax == py == expected
