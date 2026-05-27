# AI Agent Workflow

Use this file when working with Codex, Gemini, Copilot, Cursor, or another coding assistant.

## What Matters Most

The current active objective is to improve the RL training stack under `rl_training/src/`, especially behavior cloning and reward shaping for the target + ship-bucket action space.

The broader project objective is still to improve the Orbit Wars Kaggle bot without breaking the submission contract.

The strongest known heuristic is frozen at:

```text
versions/kaggle700_current_heuristic/
```

Any heuristic change should be compared against that version.

## Directory Ownership

| Path | Purpose | Agent Guidance |
| --- | --- | --- |
| `main.py` | Kaggle entrypoint | Keep tiny; expose `agent(obs)` only. |
| `src/` | Live heuristic bot | Main place for strategy work. |
| `configs/` | Tunable heuristic config | Prefer config changes before code complexity. |
| `versions/` | Frozen snapshots | Add new snapshots; do not overwrite old ones. |
| `scripts/` | Local eval, packaging, profiling | Use for validation; keep CLI stable. |
| `docs/` | Project knowledge | Update when strategy or workflow changes. |
| `analysis/fast_kaggle_env/` | Generated optimized Kaggle env | Do not edit manually. |
| `rl_training/` | RL experiments | Current active focus. Keep experimental ML code here. |
| `rl_training/src/` | RL policy/env/features/PPO code | Active implementation target for RL work. |
| `docs/RL_TRAINING_NOTES.md` | Current RL diagnosis and next steps | Read before changing RL training behavior. |

## Recommended Development Loop

1. State the hypothesis.
2. Make one narrow change.
3. Run a smoke benchmark.
4. Run fixed-seed comparison against `versions/kaggle700_current_heuristic/main.py`.
5. Keep the change only if evidence improves.
6. If submitted to Kaggle or clearly valuable, freeze a new `versions/<name>/` snapshot and update `docs/BOT_REGISTRY.md`.

## Evidence Standard

Weak evidence:

- one game
- one seed
- only vs `random` or `noop`
- only training reward

Useful evidence:

- fixed seed range
- current bot vs frozen score-700 bot
- direct runner speed and reward
- renderable Kaggle-style replay for suspicious results
- repeated runs when variance is high

## Current RL Status

PPO training exists under `rl_training/`. The older target-only setup showed some unstable signal:

- random init vs heuristic: `1/6` wins in one sweep
- best checkpoint `ckpt_000060`: `3/6` wins in direct eval
- final checkpoint regressed

Interpretation: the model can learn something, but sparse terminal PPO is not stable enough yet.

The newer target + ship-bucket setup runs end to end, but current results are weak:

- scratch bucket PPO: `0/20` vs current/best heuristic, `1/20` vs sniper, `13/20` vs random
- BC-initialized bucket PPO: `0/20` vs current/best heuristic, `5/20` vs sniper, `13/20` vs random

Interpretation: this is not a "just train longer" problem. Terminal-only PPO, missing bucket BC labels, weak observations, and blunt credit assignment are blocking learning. See `docs/RL_TRAINING_NOTES.md`.

Recommended next RL task:

```text
Fix BC dataset builders -> add target + ship-bucket labels -> train BC -> add shaped rewards -> evaluate -> PPO fine-tune.
```

Do not start long Kaggle GPU training before behavior cloning includes bucket supervision and better rewards.

## GPU And Kaggle Handoff

The user may run heavy experiments on **Kaggle notebook GPU** instead of the local machine.

Agents should **say explicitly when GPU is the better option**, then let the user run that job on Kaggle and report back. Do not silently burn long CPU runs when GPU would be clearly preferable.

Use GPU / suggest Kaggle for:

- Large BC dataset builds plus multi-epoch training (especially BC v2 planet-slot models)
- PPO runs beyond smoke configs (`reinforce_bucket_ppo_v2.yaml`, BC fine-tune, 200+ updates)
- Full baseline training such as `versions/rl_baseline/kaggle_train.yaml` (`total_updates: 2000`)
- Jobs expected to take more than ~15-30 minutes on local CPU, or that already proved slow in prior sessions

Keep local CPU for:

- Heuristic benchmarks, packaging, and short smoke tests
- Small sanity checks before handing off a longer GPU run

When recommending Kaggle, provide copy-paste commands with `--device cuda` (or YAML `device: cuda`), note expected runtime, and keep a minimal local smoke command when feasible.

Submission code in `main.py` stays CPU-only. GPU guidance applies under `rl_training/` and frozen RL snapshots such as `versions/rl_baseline/`.

## Common Commands

From project root:

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a main.py \
  --agent-b noop \
  --games 3 \
  --seed-start 20 \
  --episode-steps 200 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Compare current heuristic to frozen best:

```bash
conda run -n ml python scripts/bench_direct.py \
  --agent-a main.py \
  --agent-b versions/kaggle700_current_heuristic/main.py \
  --games 20 \
  --seed-start 1000 \
  --episode-steps 500 \
  --kaggle-env-root analysis/fast_kaggle_env
```

Package a submission:

```bash
conda run -n ml python scripts/package_submission.py
```

From `rl_training/`, evaluate an RL checkpoint:

```bash
conda run -n ml python eval_vs_current_heuristic.py \
  --games 20 \
  --seed-start 3000 \
  --episode-steps 200 \
  --config local_value_train.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/local_value_ppo_vs_heuristic/ckpt_000060.pt \
  --device cpu
```

Generic RL policy evaluation against `heuristic`, `sniper`, or `random`:

```bash
conda run -n ml python eval_policy.py \
  --config reinforce_bucket_ppo.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/reinforce_bucket_ppo_vs_heuristic/ckpt_last.pt \
  --baseline heuristic \
  --games 20 \
  --seed-start 4000 \
  --episode-steps 200 \
  --device cpu \
  --deterministic
```

## Version Freezing

Create a new milestone version:

```bash
mkdir -p versions/<id>/src versions/<id>/configs
cp main.py versions/<id>/
cp src/*.py versions/<id>/src/
cp configs/bot_config.json versions/<id>/configs/
```

Then update:

- `docs/BOT_REGISTRY.md`
- `versions/README.md`

## Agent-Specific Files

- Codex and other generic agents: `AGENTS.md`
- Gemini: `GEMINI.md`
- Cursor: `.cursor/rules/orbit-wars.mdc`
- GitHub Copilot: `.github/copilot-instructions.md`
