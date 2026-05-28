# Orbit Wars Agent Instructions

This repository is an **RL-first** Orbit Wars project with two stacks:

- Active: `rl_training_jax/` (JAX/Flax Transformer PPO).
- Legacy: `rl_training/` (PyTorch PPO + BC).

Heuristic bots are frozen under `versions/` and used only as opponents/benchmarks.

## Project layout

| Path | Purpose |
|------|---------|
| `rl_training_jax/` | Active JAX PPO training, eval/export, configs |
| `rl_training/` | Legacy PyTorch PPO + BC stack (reference) |
| `versions/kaggle700_current_heuristic/` | Primary heuristic opponent (~700 LB) |
| `analysis/fast_kaggle_env/` | Fast local simulator |
| `scripts/bench_direct.py` | Head-to-head benchmarking |
| `kaggle/` | Kaggle notebook helpers and submission smoke test |
| `submission_jax/` | Kaggle submission template |

## Read first

1. `docs/GAME_RULES.md`
2. `docs/RL_STRATEGY.md`
3. `docs/JAX_PPO_PLAN.md`
4. `docs/JAX_TRAINING.md`
5. `docs/KAGGLE_GUIDE.md`

## Development focus

Primary focus is `rl_training_jax/src/`:

- `orbit_wars/features_jax.py`
- `policy.py`
- `orbit_wars/decode.py`
- `orbit_wars/rollout.py`
- `ppo.py`
- `train_ppo.py`

Legacy improvements in `rl_training/src/` are allowed when explicitly requested.

Do not edit heuristic code except under `versions/` when freezing a new opponent snapshot.

## Validation

```bash
cd rl_training_jax
conda run -n ml python -m pytest tests/ -q
```

```bash
cd rl_training_jax
PYTHONPATH=src conda run -n ml python -m train_ppo \
  --config configs/smoke_transformer.yaml
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

Use Kaggle GPU for long JAX runs. Current memory-safe default for T4:

- `num_envs: 32`
- `minibatch_size: 256`

If OOM appears, reduce `minibatch_size` first. Keep local CPU for smoke/tests only.

## Submission flow

1. Train with `rl_training_jax/src/train_ppo.py`.
2. Export with `rl_training_jax/scripts/export_jax_submission.py`.
3. Validate with `kaggle/test_submission_locally.py`.
4. Upload `submission_jax.zip` to Kaggle competition.

## Versioning

Freeze meaningful bots under `versions/<name>/` and record in `docs/BOT_REGISTRY.md`. Do not overwrite existing versions.
