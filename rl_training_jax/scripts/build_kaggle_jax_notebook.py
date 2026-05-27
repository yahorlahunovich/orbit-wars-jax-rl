#!/usr/bin/env python3
"""Build minimal Kaggle GPU notebook: JAX PPO, 100 updates, training logs only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = REPO / "notebooks" / "orbit-wars-jax-ppo-kaggle.ipynb"

RL_SRC_FILES = [
    "src/config.py",
    "src/game_types.py",
    "src/geometry.py",
    "src/notebook_features.py",
    "src/features.py",
]

JAX_SRC_FILES = [
    "src/policy.py",
    "src/ppo.py",
    "src/train_kaggle.py",
]

HEURISTIC_STUB = '''\
"""Kaggle stub — heuristic planner disabled."""

from __future__ import annotations
from typing import Any


def heuristic_plan_for_target(*args: Any, **kwargs: Any) -> None:
    return None
'''

CONFIG_CELL = '''\
from pathlib import Path

NUM_ENVS = 8
ROLLOUT_STEPS = 32
TOTAL_UPDATES = 100
LOG_EVERY = 1
CHECKPOINT_EVERY = 25

yaml_text = f"""
seed: 321
run_name: jax_scratch_ppo_kaggle
save_dir: artifacts
log_every: {LOG_EVERY}
checkpoint_every: {CHECKPOINT_EVERY}
opponent: random

env:
  episode_steps: 200
  candidate_count: 49
  ship_bucket_count: 5

model:
  hidden_size: 128

ppo:
  rollout_steps: {ROLLOUT_STEPS}
  num_envs: {NUM_ENVS}
  total_updates: {TOTAL_UPDATES}
  epochs: 3
  minibatch_size: {NUM_ENVS * 128}
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  lr: 0.001
  lr_end: 0.0001
  max_grad_norm: 0.5
""".strip() + "\\n"

Path("kaggle_jax_train.yaml").write_text(yaml_text, encoding="utf-8")
print(f"config ready: {NUM_ENVS} envs x {ROLLOUT_STEPS} rollout x {TOTAL_UPDATES} updates")
'''


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source if source.endswith("\n") else source + "\n",
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if source.endswith("\n") else source + "\n",
    }


def writefile_cell(relpath: str, content: str) -> dict:
    return code_cell(f"%%writefile {relpath}\n\n{content.rstrip()}\n")


def patch_rl_imports(text: str) -> str:
    for old, new in [
        ("from .config import", "from rl_features.config import"),
        ("from .game_types import", "from rl_features.game_types import"),
        ("from .geometry import", "from rl_features.geometry import"),
        ("from .heuristic_adapter import", "from rl_features.heuristic_adapter import"),
        ("from .notebook_features import", "from rl_features.notebook_features import"),
        ("from .features import", "from rl_features.features import"),
        ("from config import", "from rl_features.config import"),
        ("from game_types import", "from rl_features.game_types import"),
        ("from geometry import", "from rl_features.geometry import"),
        ("from heuristic_adapter import", "from rl_features.heuristic_adapter import"),
        ("from notebook_features import", "from rl_features.notebook_features import"),
        ("from features import", "from rl_features.features import"),
        ("from .policy import", "from policy import"),
        ("from policy import", "from policy import"),
    ]:
        text = text.replace(old, new)
    return text


def main() -> None:
    rl_root = REPO / "rl_training"
    cells: list[dict] = []

    cells.append(md_cell(
        "# Orbit Wars JAX PPO — Kaggle GPU\n"
        "\n"
        "Scratch PPO with JAX env + Flax policy. **100 updates**, logs every step.\n"
        "\n"
        "**Before running:** enable GPU + add dataset `egorlagunovich/orbit-wars-jax-env` "
        "(must contain `rl_training_jax/src/orbit_wars/`)."
    ))

    cells.append(md_cell("## Setup"))
    cells.append(code_cell(
        "%%capture\n"
        "import subprocess, sys\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "    'jax[cuda12]', 'flax', 'optax', 'pyyaml'])\n"
    ))
    cells.append(code_cell(
        "import logging\n"
        "import os\n"
        "import sys\n"
        "\n"
        "logging.getLogger('kaggle_environments').setLevel(logging.WARNING)\n"
        "os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'\n"
        "\n"
        "JAX_DATASET = '/kaggle/input/datasets/egorlagunovich/orbit-wars-jax-env'\n"
        "JAX_SRC = f'{JAX_DATASET}/rl_training_jax/src'\n"
        "WORK = '/kaggle/working'\n"
        "\n"
        "for p in [JAX_SRC, WORK]:\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "\n"
        "import jax\n"
        "print('JAX', jax.__version__, '| devices:', jax.devices())\n"
    ))

    cells.append(md_cell("## Feature encoder + training code"))
    cells.append(code_cell("import os; os.makedirs('rl_features', exist_ok=True)\n"))
    cells.append(writefile_cell("rl_features/__init__.py", '"""PyTorch feature encoder (Kaggle notebook copy)."""\n'))

    for rel in RL_SRC_FILES:
        content = patch_rl_imports((rl_root / rel).read_text(encoding="utf-8"))
        cells.append(writefile_cell(f"rl_features/{Path(rel).name}", content))
    cells.append(writefile_cell("rl_features/heuristic_adapter.py", HEURISTIC_STUB))

    for rel in JAX_SRC_FILES:
        content = (ROOT / rel).read_text(encoding="utf-8")
        if rel.endswith("train_kaggle.py"):
            content = patch_rl_imports(content)
        cells.append(writefile_cell(Path(rel).name, content))

    cells.append(code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "WORK = Path('/kaggle/working')\n"
        "if str(WORK) not in sys.path:\n"
        "    sys.path.insert(0, str(WORK))\n"
        "\n"
        "from rl_features.config import EnvConfig\n"
        "from rl_features.features import encode_turn\n"
        "print('rl_features import ok | EnvConfig episode_steps =', EnvConfig().episode_steps)\n"
    ))

    cells.append(md_cell("## Config"))
    cells.append(code_cell(CONFIG_CELL))

    cells.append(md_cell("## Train (100 updates)"))
    cells.append(code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "WORK = Path('/kaggle/working')\n"
        "RL_FEATURES = WORK / 'rl_features'\n"
        "assert (RL_FEATURES / 'config.py').is_file(), (\n"
        "    'Missing rl_features/config.py — run the \"Feature encoder + training code\" cells first.'\n"
        ")\n"
        "work = str(WORK)\n"
        "if work not in sys.path:\n"
        "    sys.path.insert(0, work)\n"
        "\n"
        "import importlib\n"
        "import train_kaggle\n"
        "importlib.reload(train_kaggle)\n"
        "\n"
        "cfg = train_kaggle.load_config('kaggle_jax_train.yaml')\n"
        "train_kaggle.train(cfg)\n"
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
            "gpuClass": "standard",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
