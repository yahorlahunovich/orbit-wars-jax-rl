# Fast Training Environment

This project now has a faster local Orbit Wars simulation path for training and profiling.

It does **not** change the Kaggle submission bot. Kaggle still needs `main.py` plus `src/` packaged normally. The fast environment is only for local rollout generation, profiling, and future RL training.

## What Exists

| Path | Purpose |
| --- | --- |
| `analysis/fast_kaggle_env/` | Generated patched `kaggle_environments` tree. Do not edit by hand. |
| `scripts/build_fast_env.py` | Rebuilds `analysis/fast_kaggle_env/` from downloaded Kaggle source. |
| `scripts/fast_orbit_core.py` | Numba kernels for fleet collision and comet path generation. |
| `scripts/profile_env.py` | Measures `env.run` wall time and estimated agent/env split. |
| `scripts/parity_env.py` | Compares official env vs patched fast env step-by-step. |
| `scripts/direct_runner.py` | Calls Orbit Wars `interpreter` directly, bypassing Kaggle `env.run` overhead. |
| `scripts/parity_direct.py` | Compares direct runner vs Kaggle `env.run`. |
| `scripts/bench_direct.py` | Benchmarks direct runner throughput. |
| `scripts/cprofile_env.py` | Shows remaining Python bottlenecks. |

## Assumptions

- Use conda env `ml`.
- Downloaded Kaggle source is at:

```bash
/media/yahor/ADATA SE880/datasets/kaggle-environments-master
```

If that path changes, pass the new path to `--source-root`, `--baseline-env-root`, or `--kaggle-env-root`.

## Rebuild Fast Env

Run from `orbit_wars_cursor_template_v2/`:

```bash
conda run -n ml python scripts/build_fast_env.py \
  --source-root "/media/yahor/ADATA SE880/datasets/kaggle-environments-master" \
  --output-root analysis/fast_kaggle_env
```

This deletes and recreates `analysis/fast_kaggle_env/`.

## Validate Correctness

Fast env must match the official downloaded env:

```bash
conda run -n ml python scripts/parity_env.py \
  --baseline-env-root "/media/yahor/ADATA SE880/datasets/kaggle-environments-master" \
  --candidate-env-root analysis/fast_kaggle_env \
  --seed 21 \
  --episode-steps 200
```

Direct runner must match fast `env.run`:

```bash
conda run -n ml python scripts/parity_direct.py \
  --agent-a probe \
  --agent-b probe \
  --seed 0 \
  --episode-steps 180 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Also test your actual bot path:

```bash
conda run -n ml python scripts/parity_direct.py \
  --agent-a main.py \
  --agent-b noop \
  --seed 21 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

If parity fails, do not use the fast/direct runner for training until the mismatch is understood.

## Benchmark

Fast Kaggle `env.run`:

```bash
conda run -n ml python scripts/profile_env.py \
  --agent-a noop \
  --agent-b noop \
  --games 3 \
  --seed-start 20 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Direct runner:

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a noop \
  --agent-b noop \
  --games 3 \
  --seed-start 20 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Direct runner with current bot:

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a main.py \
  --agent-b noop \
  --games 3 \
  --seed-start 20 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Recent results:

| Setup | Result |
| --- | --- |
| Fast `env.run`, no-op/no-op | about `2.6 ms/step` |
| Direct runner, no-op/no-op | about `0.9 ms/step` |
| Direct runner, `main.py`/no-op | about `7.4 ms/step` |

Interpretation: the environment is now fast enough that the current heuristic bot often dominates runtime.

## Use From Python

Minimal example:

```python
import sys
from pathlib import Path

sys.path.insert(0, "analysis/fast_kaggle_env")

from scripts.direct_runner import run_direct_from_names

root = Path(".").resolve()
steps, elapsed = run_direct_from_names(
    ["main.py", "noop"],
    root=root,
    seed=21,
    episode_steps=200,
    keep_steps=True,
)

final = steps[-1]
print(elapsed, final[0].reward, final[1].reward)
```

For high-throughput RL rollouts, use `keep_steps=False` unless you need full traces.

## Caveats

- `analysis/fast_kaggle_env/` is generated. Rebuild it instead of editing files inside it.
- Numba has import/compile overhead. First process startup can be slower than steady state.
- The direct runner intentionally bypasses Kaggle schema validation, deepcopy, logging, and process wrappers.
- Direct runner is for training and local evaluation only. Do not package it for Kaggle submission.
- Always run parity checks after changing simulator code, the direct runner, or the fast kernels.

# Project Plan

## Phase 1: Stabilize Fast Simulation

Status: mostly done.

1. Keep `parity_env.py` and `parity_direct.py` as mandatory checks.
2. Add a multi-seed parity command/script, e.g. seeds `0..50`, episode steps `500`.
3. Add 4-player parity tests, because FFA may expose different combat and action-volume patterns.
4. Add a CI-style smoke command that runs compile + parity + benchmark.

## Phase 2: Reduce Bot Runtime

Current finding: `main.py` is slower than the optimized environment in direct rollouts.

Focus areas:

1. Profile `src/strategy.py` under direct runner.
2. Cache repeated distance/intercept calculations per turn.
3. Reduce candidate search from all targets/sources to a fixed top-k shortlist.
4. Avoid repeated obstacle construction in `obstacles_for_path`.
5. Add a cheap “training policy” mode for rollout collection if full heuristic is too expensive.

Success target:

```text
current main.py direct rollout: ~7.4 ms/step
target: <=2.0 ms/step
```

## Phase 3: RL Data Interface

Build a stable RL-facing API around the direct runner.

1. `reset(seed) -> observation`
2. `step(actions) -> observation, reward, done, info`
3. Fixed candidate action builder:
   - per owned planet
   - top-k target candidates
   - legal mask
   - no-op candidate
4. Feature encoder:
   - source planet features
   - candidate target features
   - global economy/military features
5. Deterministic action decoder:
   - policy chooses target candidate
   - heuristic computes ship count and exact angle

## Phase 4: Learning Strategy

Avoid full continuous-action deep RL at first.

Recommended sequence:

1. Behavior cloning from current heuristic:
   - train policy to imitate `main.py` target choices
   - validates features/model/export path cheaply
2. PPO against weak opponents:
   - random
   - starter/sniper
   - frozen heuristic versions
3. Self-play with opponent snapshots:
   - latest policy
   - older checkpoints
   - heuristic fallback
4. Hybrid inference:
   - RL ranks candidates
   - deterministic safety layer rejects bad launches
   - heuristic fallback when model confidence/action mask is poor

## Phase 5: Evaluation Discipline

Every candidate model/bot should be tested with:

```bash
python scripts/evaluate.py --agent-a main.py --agent-b random --games 50
```

Then against saved versions:

```bash
python scripts/evaluate.py \
  --agent-a path/to/new/main.py \
  --agent-b versions/baseline_kaggle526/main.py \
  --games 200
```

Keep a row in `docs/BOT_REGISTRY.md` for every submitted or serious checkpoint.

## Immediate Next Task

Profile and optimize `src/strategy.py` under the direct runner.

Suggested command to create next:

```bash
conda run -n ml python scripts/cprofile_direct.py \
  --agent-a main.py \
  --agent-b noop \
  --games 1 \
  --seed-start 21 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Expected outcome: identify the top 3 hot functions in the bot and reduce repeated geometry work.
