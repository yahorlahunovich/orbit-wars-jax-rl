"""Load the frozen kaggle700 heuristic and produce padded action tensors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .constants import MAX_MOVES_PER_PLAYER
from .convert import state_to_observation_dict
from .state import OrbitWarsState


def default_heuristic_path() -> Path:
    """Resolve `versions/kaggle700_current_heuristic/main.py` from repo root."""
    here = Path(__file__).resolve()
    for root in here.parents:
        candidate = root / "versions" / "kaggle700_current_heuristic" / "main.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find versions/kaggle700_current_heuristic/main.py. "
        "Include `versions/` in your Kaggle dataset or git clone."
    )


def load_heuristic_agent(path: Path | None = None) -> Callable[[Any], list]:
    """Import and return the heuristic `agent(obs)` function."""
    bot_path = (path or default_heuristic_path()).resolve()
    heur_root = bot_path.parent
    if str(heur_root) not in sys.path:
        sys.path.insert(0, str(heur_root))
    spec = importlib.util.spec_from_file_location("heuristic_main", bot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load heuristic from {bot_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def pack_moves_list(moves: list[list[float | int]]) -> tuple[np.ndarray, np.ndarray]:
    """Pack Kaggle-style moves into `(MAX_MOVES, 3)` + mask."""
    actions = np.zeros((MAX_MOVES_PER_PLAYER, 3), dtype=np.float32)
    mask = np.zeros((MAX_MOVES_PER_PLAYER,), dtype=np.float32)
    for i, row in enumerate(moves[:MAX_MOVES_PER_PLAYER]):
        actions[i, 0] = float(row[0])
        actions[i, 1] = float(row[1])
        actions[i, 2] = float(row[2])
        mask[i] = 1.0
    return actions, mask


def heuristic_actions_for_state(state: OrbitWarsState, player: int, agent) -> tuple[np.ndarray, np.ndarray]:
    obs = state_to_observation_dict(state, player=int(player))
    moves = agent(obs)
    return pack_moves_list(moves)


def batched_heuristic_actions(
    states: OrbitWarsState,
    opponent_players: np.ndarray,
    agent,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build padded `(B, M, 3)` action tensors for player 0 and player 1.

    `opponent_players[i]` is the heuristic seat for env *i* (0 or 1).
    Non-heuristic seats are zero with mask 0.
    """
    import jax.tree_util as tu

    n = int(states.step.shape[0])
    a0 = np.zeros((n, MAX_MOVES_PER_PLAYER, 3), dtype=np.float32)
    m0 = np.zeros((n, MAX_MOVES_PER_PLAYER), dtype=np.float32)
    a1 = np.zeros((n, MAX_MOVES_PER_PLAYER, 3), dtype=np.float32)
    m1 = np.zeros((n, MAX_MOVES_PER_PLAYER), dtype=np.float32)

    for i in range(n):
        single = tu.tree_map(lambda x, i=i: x[i], states)
        opp = int(opponent_players[i])
        act, msk = heuristic_actions_for_state(single, opp, agent)
        if opp == 0:
            a0[i] = act
            m0[i] = msk
        else:
            a1[i] = act
            m1[i] = msk

    return jnp.asarray(a0), jnp.asarray(m0), jnp.asarray(a1), jnp.asarray(m1)
