#!/usr/bin/env python3
"""Build Kaggle GPU notebook from current rl_training sources."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "orbit-wars-reinforcement-learning-tutorial-kaggle.ipynb"

SRC_FILES = [
    "src/__init__.py",
    "src/heuristic_adapter.py",
    "src/config.py",
    "src/game_types.py",
    "src/features.py",
    "src/policy.py",
    "src/ppo.py",
    "src/opponents.py",
    "src/env.py",
    "src/train.py",
]

HEURISTIC_FILES = [
    "heuristic_bot/src/__init__.py",
    "heuristic_bot/src/bot.py",
    "heuristic_bot/src/constants.py",
    "heuristic_bot/src/game.py",
    "heuristic_bot/src/geometry.py",
    "heuristic_bot/src/strategy.py",
    "heuristic_bot/configs/bot_config.json",
]

KAGGLE_HEURISTIC_ADAPTER = ROOT / "kaggle" / "heuristic_adapter.py"
if not KAGGLE_HEURISTIC_ADAPTER.exists():
    KAGGLE_HEURISTIC_ADAPTER = None


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
    body = content.rstrip("\n") + "\n"
    return code_cell(f"%%writefile {relpath}\n\n{body}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_scaling_cell() -> dict:
    return code_cell(
        '''# Scale rollout/minibatch with parallel env count.
# Re-run this cell before training when changing NUM_ENVS.

from pathlib import Path

NUM_ENVS = 16
ROLLOUT_STEPS = 32
MINIBATCH_SIZE = NUM_ENVS * 128  # 2048 when NUM_ENVS=16
TOTAL_UPDATES = 100  # smoke test; set 500-2000 for long training
CHECKPOINT_EVERY = 25
LOG_EVERY = 5

yaml_text = f"""
seed: 321
run_name: scratch_ppo_kaggle_v2
device: cuda
save_dir: artifacts
checkpoint_every: {CHECKPOINT_EVERY}
log_every: {LOG_EVERY}
opponent: heuristic
self_play_update_interval: 50
self_play_deterministic: false
alternate_player_sides: true

env:
  episode_steps: 200
  candidate_count: 49
  ship_bucket_count: 5
  use_fast_env: false
  kaggle_env_root: ""
  use_heuristic_planner: false

model:
  hidden_size: 128

ppo:
  rollout_steps: {ROLLOUT_STEPS}
  num_envs: {NUM_ENVS}
  total_updates: {TOTAL_UPDATES}
  epochs: 3
  minibatch_size: {MINIBATCH_SIZE}
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  lr: 0.001
  lr_end: 0.0001
  max_grad_norm: 0.5
""".strip() + "\\n"

Path("kaggle_train.yaml").write_text(yaml_text, encoding="utf-8")
print(f"Wrote kaggle_train.yaml: num_envs={NUM_ENVS}, rollout_steps={ROLLOUT_STEPS}, "
      f"minibatch_size={MINIBATCH_SIZE}, total_updates={TOTAL_UPDATES}")
'''
    )


def main() -> None:
    cells: list[dict] = []

    cells.append(md_cell(
        "# Orbit Wars Scratch PPO (Kaggle GPU)\n"
        "\n"
        "Copy of the RL tutorial notebook, updated for current `rl_training`:\n"
        "\n"
        "- **Planet-slot targets** (49 slots: no-op + 48 planets)\n"
        "- **Ship fraction buckets** (25/50/75/100% surplus + exact mission size)\n"
        "- **Bucket features** for the ship head\n"
        "- **GAE**, shaped step rewards + win/loss terminal reward\n"
        "- **Higher LR** (`1e-3 → 1e-4`)\n"
        "- **PPO diagnostics**: explained variance, approx KL, clip fraction\n"
        "- **Scratch PPO** (no BC init)\n"
        "- Bundled score-700 heuristic opponent under `heuristic_bot/`\n"
        "\n"
        "Enable **GPU** before running. Default smoke run: **100 updates**."
    ))

    cells.append(md_cell(
        "## Action space\n"
        "\n"
        "Per owned planet each turn:\n"
        "\n"
        "1. **Target head** — pick planet slot (`0 = no-op`, `1..48 = planet id`)\n"
        "2. **Ship bucket head** — pick fleet size bin for that target\n"
        "\n"
        "Launch angle comes from geometry toward the target (no heuristic planner by default)."
    ))

    cells.append(md_cell(
        "## Scaling knobs\n"
        "\n"
        "Edit `NUM_ENVS` in the scaling cell. We set:\n"
        "\n"
        "- `rollout_steps = 32`\n"
        "- `minibatch_size = NUM_ENVS * 128` (2048 at 16 envs)\n"
        "- `total_updates = 100` for smoke; bump to 500–2000 for long training"
    ))

    cells.append(md_cell("## Setup"))
    cells.append(code_cell('%%capture\n!pip install --upgrade "kaggle-environments>=1.28.0" pyyaml\n'))
    cells.append(code_cell("!mkdir -p src heuristic_bot/src heuristic_bot/configs artifacts\n"))

    cells.append(md_cell("## Write training config"))
    cells.append(build_scaling_cell())

    cells.append(md_cell("## Write training code (from current `rl_training/src`)"))

    heuristic_src = REPO / "versions" / "kaggle700_current_heuristic" / "src"
    heuristic_cfg = REPO / "versions" / "kaggle700_current_heuristic" / "configs" / "bot_config.json"

    # Kaggle heuristic adapter prefers heuristic_bot/ bundle
    adapter_path = ROOT / "orbit-wars-rl-bucket-training-kaggle.ipynb"
    adapter_content = None
    if adapter_path.exists():
        nb = json.loads(adapter_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            src = "".join(cell.get("source", []))
            if "%%writefile src/heuristic_adapter.py" in src:
                adapter_content = src.split("%%writefile src/heuristic_adapter.py\n\n", 1)[1]
                break

    for rel in SRC_FILES:
        if rel == "src/heuristic_adapter.py" and adapter_content:
            cells.append(writefile_cell(rel, adapter_content))
            continue
        path = ROOT / rel
        cells.append(writefile_cell(rel, read_text(path)))

    cells.append(md_cell("## Bundle score-700 heuristic opponent"))
    for rel in HEURISTIC_FILES:
        if rel.endswith("bot_config.json"):
            path = heuristic_cfg
        else:
            path = heuristic_src / Path(rel).name
        cells.append(writefile_cell(rel, read_text(path)))

    eval_path = ROOT / "kaggle" / "eval_kaggle.py"
    if eval_path.exists():
        cells.append(md_cell("## Write evaluation script"))
        eval_src = read_text(eval_path)
        eval_src = eval_src.replace(
            'default="artifacts/reinforce_bucket_ppo_v2_kaggle/ckpt_002000.pt"',
            'default="artifacts/scratch_ppo_kaggle_v2/ckpt_000100.pt"',
        )
        cells.append(writefile_cell("eval_kaggle.py", eval_src))

    cells.append(md_cell(
        "## Train scratch PPO (100 updates smoke test)\n"
        "\n"
        "Watch for non-zero `approx_kl`, rising `explained_variance`, and non-trivial `clip_fraction` "
        "before starting a long run."
    ))
    cells.append(code_cell(
        "import sys\n"
        "!{sys.executable} -m src.train --config kaggle_train.yaml --no-bc-init\n"
    ))

    cells.append(md_cell("## Evaluate checkpoint vs bundled heuristic"))
    cells.append(code_cell(
        "import sys\n"
        "!{sys.executable} eval_kaggle.py \\\n"
        "  --config kaggle_train.yaml \\\n"
        "  --checkpoint artifacts/scratch_ppo_kaggle_v2/ckpt_000100.pt \\\n"
        "  --baseline heuristic \\\n"
        "  --games 10 \\\n"
        "  --seed 3000 \\\n"
        "  --device cuda \\\n"
        "  --deterministic\n"
    ))

    cells.append(md_cell(
        "## Long training\n"
        "\n"
        "If the smoke run looks healthy, set `TOTAL_UPDATES = 2000` in the scaling cell, "
        "re-run scaling + training cells, and leave the notebook running."
    ))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
