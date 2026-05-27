from __future__ import annotations

import sys
from pathlib import Path


def setup_rl_script_paths() -> tuple[Path, Path]:
    """Add repo root, rl_training, scripts, and fast env to sys.path."""
    rl_root = Path(__file__).resolve().parents[1]
    repo_root = rl_root.parent
    fast_env = repo_root / "analysis" / "fast_kaggle_env"
    for path in (rl_root, repo_root / "scripts", fast_env):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root, rl_root
