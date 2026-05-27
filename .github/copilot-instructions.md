# GitHub Copilot Instructions

This repository contains a Kaggle Orbit Wars bot.

## Repository Layout

- `main.py` - Kaggle entrypoint exposing `agent(obs)`.
- `src/` - live heuristic bot implementation.
- `configs/bot_config.json` - heuristic parameters.
- `versions/kaggle700_current_heuristic/` - frozen best heuristic snapshot, Kaggle score 700.
- `rl_training/` - PPO/RL experiments and evaluation scripts.
- `analysis/fast_kaggle_env/` - optimized local Kaggle environment; do not edit manually.
- `docs/` - game rules, architecture, registry, and development workflow.

## Completion Guidance

When suggesting code:

- Preserve `agent(obs)` and the move format `[[planet_id, angle_radians, ships], ...]`.
- Prefer small deterministic heuristic changes.
- Keep geometry helpers in `src/geometry.py`.
- Keep observation parsing in `src/game.py`.
- Keep strategy logic in `src/strategy.py`.
- Keep tunables in `configs/bot_config.json`.
- Do not add heavy runtime dependencies.
- Do not print from submission agent code.
- Do not modify frozen versions except when explicitly creating a new version.

## Preferred Benchmarks

Use the `ml` conda environment:

```bash
conda run -n ml python scripts/bench_direct.py --agent-a main.py --agent-b noop --games 3 --episode-steps 200 --kaggle-env-root analysis/fast_kaggle_env
```

Compare improvements to the frozen best:

```bash
conda run -n ml python scripts/bench_direct.py --agent-a main.py --agent-b versions/kaggle700_current_heuristic/main.py --games 20 --seed-start 1000 --episode-steps 500 --kaggle-env-root analysis/fast_kaggle_env
```

## RL Guidance

The PPO work in `rl_training/` is experimental. The next recommended RL step is behavior cloning from the current heuristic, then PPO fine-tuning. Avoid assuming longer PPO training alone will improve results.
