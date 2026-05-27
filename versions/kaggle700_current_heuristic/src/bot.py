from __future__ import annotations

from typing import Any

from src.game import parse_state
from src.strategy import decide_moves


def agent(obs: Any) -> list[list[int | float]]:
    try:
        state = parse_state(obs)
        return decide_moves(state)
    except Exception:
        return []
