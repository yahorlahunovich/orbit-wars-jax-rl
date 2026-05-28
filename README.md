# Orbit Wars — RL Project

Reinforcement-learning workspace for [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars).

This repo now has two training stacks:

- `rl_training/`: legacy PyTorch PPO + BC pipeline (still useful as reference).
- `rl_training_jax/`: active JAX/Flax Transformer PPO pipeline (current focus).

Heuristic bots remain frozen under `versions/` and are used only as opponents/benchmarks.

## Layout

```text
orbit_wars_cursor_template_v2/
  rl_training/              Legacy PPO + BC training stack (PyTorch)
  rl_training_jax/          Active JAX Transformer PPO stack
  versions/                 Frozen heuristic opponents (kaggle700, baseline526, ...)
  scripts/                  Shared bench/direct-runner utilities
  kaggle/                   Kaggle notebook helpers and submission smoke test
  submission_jax/           Kaggle submission template (export target)
  docs/                     Strategy, plans, and training guides
```

## JAX quick start (recommended)

```bash
cd rl_training_jax

# Run tests
conda run -n ml python -m pytest tests/ -q

# CPU smoke run (20 updates)
PYTHONPATH=src conda run -n ml python -m train_ppo \
  --config configs/smoke_transformer.yaml
```

## Kaggle training workflow

1. Follow `docs/KAGGLE_GUIDE.md`.
2. Train on Kaggle GPU using `configs/transformer_selfplay.yaml`.
3. Export a competition-ready zip:

```bash
cd rl_training_jax
conda run -n ml python scripts/export_jax_submission.py \
  --checkpoint artifacts/jax_ppo_transformer/ckpt_last.npz \
  --config configs/transformer_selfplay.yaml \
  --output ../submission_jax.zip
```

4. Validate locally before upload:

```bash
cd ..
conda run -n ml python kaggle/test_submission_locally.py --submission submission_jax.zip
```

## Current JAX status

- Completed: feature encoder, transformer policy, geometry decoder, masked rollout sampling, PPO+GAE, and smoke training pipeline.
- Current focus: faster on-device rollout and stronger long-run Kaggle training/eval.

## Heuristic benchmark (reference)

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a versions/kaggle700_current_heuristic/main.py \
  --agent-b noop \
  --games 3 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

See `AGENTS.md` for coding-agent instructions and `docs/KAGGLE_GUIDE.md` for deployment steps.
