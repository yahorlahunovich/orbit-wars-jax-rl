#!/usr/bin/env python3
"""Build minimal Kaggle GPU notebook: JAX PPO, self-contained."""

from __future__ import annotations

import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = REPO / "notebooks" / "orbit-wars-jax-ppo-kaggle.ipynb"

ORBIT_WARS_FILES = [
    "__init__.py",
    "comet.py",
    "constants.py",
    "convert.py",
    "decode.py",
    "env.py",
    "features_jax.py",
    "geometry.py",
    "heuristic_opponent.py",
    "producer.py",
    "reference.py",
    "reset.py",
    "rollout.py",
    "state.py",
    "step.py",
]

JAX_SRC_FILES = [
    "policy.py",
    "ppo.py",
    "train_ppo.py",
]

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
    return code_cell(f"%%writefile {relpath}\n{content.rstrip()}\n")

def main() -> None:
    cells: list[dict] = []

    cells.append(md_cell(
        "# Orbit Wars JAX PPO — Kaggle GPU\n"
        "\n"
        "100% self-contained scratch PPO with JAX env + Flax policy.\n"
        "**Before running:** enable GPU Accelerator (T4x2 or P100) and Internet."
    ))

    cells.append(md_cell("## Setup"))
    cells.append(code_cell(
        "%%capture\n"
        "import subprocess, sys\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "    'jax[cuda12]', 'flax', 'optax', 'pyyaml', 'numba'])\n"
    ))
    
    cells.append(code_cell("import os\nos.makedirs('orbit_wars', exist_ok=True)\nos.makedirs('configs', exist_ok=True)\n"))

    cells.append(md_cell("## Environment and Policy source code"))
    
    # orbit_wars module
    for fname in ORBIT_WARS_FILES:
        content = (ROOT / "src" / "orbit_wars" / fname).read_text(encoding="utf-8")
        cells.append(writefile_cell(f"orbit_wars/{fname}", content))

    # Top-level src files
    for fname in JAX_SRC_FILES:
        content = (ROOT / "src" / fname).read_text(encoding="utf-8")
        cells.append(writefile_cell(fname, content))

    cells.append(md_cell("## Config"))
    
    # Include transformer_selfplay config inline
    config_content = (ROOT / "configs" / "transformer_selfplay.yaml").read_text(encoding="utf-8")
    cells.append(writefile_cell("configs/transformer_selfplay.yaml", config_content))

    cells.append(md_cell("## Start Training"))
    cells.append(code_cell(
        "import os\n"
        "os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'\n"
        "\n"
        "!python train_ppo.py --config configs/transformer_selfplay.yaml\n"
    ))

    cells.append(md_cell("## Export Submission"))
    export_content = (ROOT / "scripts" / "export_jax_submission.py").read_text(encoding="utf-8")
    # Fix import paths in export script because it will be run in the same dir
    export_content = export_content.replace(
        'sys.path.insert(0, str(ROOT / "src"))',
        'sys.path.insert(0, ".")'
    ).replace(
        'submission_dir = ROOT.parent / "submission_jax"',
        'submission_dir = Path("submission_jax")'
    ).replace(
        'src_pkg = ROOT / "src" / "orbit_wars"',
        'src_pkg = Path("orbit_wars")'
    ).replace(
        'shutil.copy2(ROOT / "src" / "policy.py", src_dir / "policy.py")',
        'shutil.copy2("policy.py", src_dir / "policy.py")'
    )

    cells.append(writefile_cell("export_jax_submission.py", export_content))
    cells.append(code_cell(
        "!python export_jax_submission.py \\\n"
        "    --checkpoint artifacts/jax_ppo_transformer/ckpt_last.npz \\\n"
        "    --config configs/transformer_selfplay.yaml \\\n"
        "    --output submission_jax.zip\n"
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
