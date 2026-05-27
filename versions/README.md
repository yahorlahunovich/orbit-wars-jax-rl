# Saved bot versions

Frozen opponents and historical heuristics. Use these paths with `scripts/evaluate.py` or `scripts/bench_direct.py`.

## Compare two versions

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a versions/kaggle700_current_heuristic/main.py \
  --agent-b versions/baseline_kaggle526/main.py \
  --games 20 \
  --seed-start 1000 \
  --episode-steps 500 \
  --kaggle-env-root analysis/fast_kaggle_env
```

## Registered versions

- **`baseline_kaggle526/`** — early baseline (~526 LB)
- **`kaggle700_current_heuristic/`** — primary RL training opponent (~700 LB)

Record new snapshots in `docs/BOT_REGISTRY.md`.

## Add a new frozen heuristic

```bash
mkdir -p versions/<id>/src versions/<id>/configs
cp path/to/main.py versions/<id>/
cp path/to/src/*.py versions/<id>/src/
cp path/to/configs/bot_config.json versions/<id>/configs/
```

Update `docs/BOT_REGISTRY.md`.
