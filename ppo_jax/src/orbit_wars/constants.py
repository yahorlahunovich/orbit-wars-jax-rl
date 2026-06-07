"""Orbit Wars simulation constants (match official Kaggle env)."""

from __future__ import annotations

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
DEFAULT_SHIP_SPEED = 6.0
DEFAULT_EPISODE_STEPS = 500
MIN_LAUNCH_SHIPS = 5

# Decoding / Planning constants
SUN_PATH_MARGIN = 1.5
PATH_PLANET_MARGIN = 1.0
INTERCEPT_ITERATIONS = 5
BUCKET_COUNT = 4

# Padded simulation limits for JIT-friendly arrays.
MAX_PLANETS = 96
MAX_FLEETS = 256
MAX_COMET_GROUPS = 8
MAX_COMET_PATH_LEN = 64
MAX_COMET_PLANETS = MAX_COMET_GROUPS * 4
MAX_MOVES_PER_PLAYER = 48
NUM_PLAYERS = 2

PLANET_COLS = 8  # id, owner, x, y, radius, ships, production, active
FLEET_COLS = 8  # id, owner, x, y, angle, from_planet_id, ships, active
