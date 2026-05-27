# RL Training

PPO + behavior cloning for Orbit Wars with planet-slot targets and ship buckets.

## Structure

```text
rl_training/
  configs/           Training YAML configs
  scripts/           BC builders, eval, diagnostics
  src/               Core env, features, policy, PPO
  artifacts/         Checkpoints and BC datasets (generated)
```

### Core modules

| Module | Role |
|--------|------|
| `src/features.py` | Observation encoder (self / candidate / global) |
| `src/notebook_features.py` | Features ported from top-score notebooks |
| `src/policy.py` | Target + bucket heads + global value head |
| `src/train.py` | PPO training loop |
| `src/heuristic_adapter.py` | Loads `versions/kaggle700_current_heuristic` as opponent |
| `src/eval_utils.py` | Checkpoint load + move decoding for eval |

### Configs

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Default BC / short runs |
| `configs/smoke_ppo.yaml` | Fast PPO sanity check vs random |
| `configs/ppo_scratch.yaml` | Scratch PPO vs heuristic |
| `configs/ppo_bc_finetune.yaml` | BC-init PPO fine-tune |
| `configs/ppo_100_terminal.yaml` | 100-update terminal-reward experiment |
| `configs/ppo_100_selfplay.yaml` | 100-update self-play experiment |
| `configs/kaggle_train.yaml` | Long GPU run template |

## Commands

Train PPO from scratch:

```bash
cd rl_training
conda run -n ml python -m src.train --config configs/ppo_scratch.yaml --no-bc-init
```

Build BC dataset and train:

```bash
conda run -n ml python scripts/build_bc_dataset.py --output artifacts/bc/top_players_bc.npz
conda run -n ml python scripts/train_bc.py --dataset artifacts/bc/top_players_bc.npz
```

Evaluate policy:

```bash
conda run -n ml python scripts/eval_policy.py \
  --config configs/ppo_scratch.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/<run>/ckpt_last.pt \
  --baseline heuristic \
  --games 20
```

Diagnose value signal:

```bash
conda run -n ml python scripts/diagnose_value_signal.py --games 20
```

## Feature dims (current)

- Global: **16** (LB 1000+ value-state vector)
- Self: **22** (source planet + threat / centrality)
- Candidate: **28** (Proto score + travel / capture signals)

Old checkpoints and BC datasets are incompatible after feature changes — regenerate BC data before BC-init PPO.

## Notes

- Rewards: terminal **+1 / -1 / 0** only (win / loss / draw).
- Heuristic opponent loads directly from `versions/kaggle700_current_heuristic/` — no runtime copy under `rl_training/`.
- GPU-worthy jobs: long BC, 1000+ PPO updates — run on Kaggle with `--device cuda`.
