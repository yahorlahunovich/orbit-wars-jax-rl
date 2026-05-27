# Orbit Wars Agent Instructions

This repository is an **RL-first** Orbit Wars project. Active development lives in `rl_training/`. Heuristic bots are frozen under `versions/` and used as opponents or benchmarks only.

## Project layout

| Path | Purpose |
|------|---------|
| `rl_training/` | PPO + BC training, eval, configs |
| `versions/kaggle700_current_heuristic/` | Primary heuristic opponent (~700 LB) |
| `analysis/fast_kaggle_env/` | Fast local simulator |
| `scripts/bench_direct.py` | Head-to-head benchmarking |
| `notebooks/` | Reference high-score notebooks (not runtime code) |

## Read first

1. `docs/GAME_RULES.md`
2. `docs/RL_STRATEGY.md`
3. `docs/RL_TRAINING_NOTES.md`
4. `docs/FAST_TRAINING_ENV.md`
5. `docs/AI_AGENT_WORKFLOW.md`

## Development focus

Improve `rl_training/src/` — especially observations (`features.py`, `notebook_features.py`), PPO loop (`train.py`, `ppo.py`), and BC pipeline (`scripts/build_bc_dataset.py`, `scripts/train_bc.py`).

Do not edit heuristic code except under `versions/` when freezing a new opponent snapshot.

## Validation

```bash
cd rl_training
conda run -n ml python -m src.train --config configs/smoke_ppo.yaml --no-bc-init
```

```bash
cd rl_training
conda run -n ml python scripts/eval_policy.py \
  --config configs/ppo_scratch.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/<run>/ckpt_last.pt \
  --baseline heuristic \
  --games 20 \
  --device cpu
```

Benchmark heuristic opponent:

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a versions/kaggle700_current_heuristic/main.py \
  --agent-b noop \
  --games 3 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

## GPU guidance

Use Kaggle GPU for long BC or 1000+ PPO updates. Give copy-paste commands with `--device cuda`. Keep local CPU for smoke tests and eval.

## Versioning

Freeze meaningful bots under `versions/<name>/` and record in `docs/BOT_REGISTRY.md`. Do not overwrite existing versions.
