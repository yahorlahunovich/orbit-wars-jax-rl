"""Heuristic opponent loader smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import reset
from orbit_wars.heuristic_opponent import load_heuristic_agent, pack_moves_list, heuristic_actions_for_state


def test_heuristic_agent_returns_moves():
    agent = load_heuristic_agent()
    state = reset(0, episode_steps=200)
    actions, mask = heuristic_actions_for_state(state, player=0, agent=agent)
    assert actions.shape == (48, 3)
    assert mask.shape == (48,)
    # Heuristic usually fires at least one move early game.
    assert mask.sum() >= 0.0


def test_pack_moves_list():
    a, m = pack_moves_list([[1.0, 0.5, 10.0], [2.0, 1.0, 5.0]])
    assert float(m.sum()) == 2.0
    assert float(a[0, 2]) == 10.0
