import json
from pathlib import Path
import os

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

def main():
    root_dir = Path(__file__).resolve().parent
    out_file = root_dir / "kaggle_jax_ppo.ipynb"
    
    cells = []
    
    cells.append(md_cell(
        "# Pure JAX PPO for Orbit Wars\n"
        "\n"
        "Self-contained notebook for Kaggle GPU (e.g. 2x T4).\n"
        "Runs JAX-based environment and purejaxrl-based PPO training."
    ))
    
    cells.append(md_cell("## Setup Packages and Directories"))
    cells.append(code_cell(
        "!pip install -q distrax gymnax\n"
        "import os\n"
        "os.makedirs('src/orbit_wars', exist_ok=True)\n"
    ))
    
    cells.append(md_cell("## Environment and Policy Source Code"))
    
    # Write orbit_wars files
    ow_dir = root_dir / "src" / "orbit_wars"
    for file_path in ow_dir.glob("*.py"):
        rel_path = f"src/orbit_wars/{file_path.name}"
        cells.append(writefile_cell(rel_path, file_path.read_text()))
        
    # Write src files
    src_dir = root_dir / "src"
    for file_path in src_dir.glob("*.py"):
        rel_path = f"src/{file_path.name}"
        cells.append(writefile_cell(rel_path, file_path.read_text()))
        
    # Write train.py
    cells.append(writefile_cell("train.py", (root_dir / "train.py").read_text()))
    
    cells.append(md_cell("## Configuration"))
    
    # 1 epoch, 100 updates config
    config_yaml = """
env:
  num_envs: 4
  rollout_steps: 128
  episode_steps: 500
  ship_speed: 6.0
model:
  d_model: 96
  num_heads: 4
  num_layers: 3
  bucket_count: 4
ppo:
  total_updates: 100
  train_pi_iters: 1
  minibatch_size: 128
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  pi_lr: 0.00025
"""
    cells.append(writefile_cell("kaggle_cfg.yaml", config_yaml))
    
    cells.append(md_cell("## Train"))
    cells.append(code_cell(
        "import os\n"
        "# Optionally set JAX to use specific GPUs\n"
        "# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'\n"
        "!python train.py --config kaggle_cfg.yaml\n"
    ))
    
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }
    
    with open(out_file, "w") as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Successfully generated notebook at {out_file}")

if __name__ == "__main__":
    main()
