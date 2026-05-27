# Orbit Wars — RL Project

Reinforcement-learning workspace for [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars). The active codebase lives under `rl_training/`. Heuristic bots are frozen under `versions/` and used only as opponents or benchmarks.

## Layout

```text
orbit_wars_cursor_template_v2/
  rl_training/          Active PPO + BC training stack
  versions/             Frozen heuristic opponents (kaggle700, baseline526, …)
  analysis/fast_kaggle_env/   Fast local simulator for rollouts
  scripts/              Shared bench/direct-runner utilities
  notebooks/            Reference high-score agent notebooks (read-only)
  docs/                 Game rules, RL strategy, training notes
```

## Quick start

```bash
cd rl_training

# Smoke PPO (scratch, no BC)
conda run -n ml python -m src.train --config configs/smoke_ppo.yaml --no-bc-init

# Evaluate checkpoint vs heuristic opponent
conda run -n ml python scripts/eval_policy.py \
  --config configs/ppo_scratch.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/<run>/ckpt_last.pt \
  --baseline heuristic \
  --games 20 \
  --device cpu
```

## Opponents

| Opponent | Source |
|----------|--------|
| `heuristic` | `versions/kaggle700_current_heuristic/` (score ~700) |
| `random` | Kaggle built-in random agent |
| `self` | Snapshot of current RL policy |

## Heuristic benchmarks

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a versions/kaggle700_current_heuristic/main.py \
  --agent-b noop \
  --games 3 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

See `rl_training/README.md` for full training workflow and `AGENTS.md` for agent instructions.
# orbit-wars-jax-rl
