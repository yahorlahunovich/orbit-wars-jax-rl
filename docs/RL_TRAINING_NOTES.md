# RL Training Notes

This document records the current RL state and the concrete weak points found in `rl_training/src/`.

## Current Focus

The RL pipeline was overhauled on 2026-05-21 to fix five critical issues that blocked learning. The pipeline now runs end to end with correct rewards, shaped intermediate signals, GAE, bucket features, and joint BC supervision. The next step is to generate fresh BC data with bucket labels, train BC, and PPO-fine-tune from the BC checkpoint.

As of 2026-05-24, PPO training logs now include **explained variance**, **approx KL**, and **clip fraction**. Local scratch-PPO vs-heuristic sweeps showed the old default LR (`3e-4`) was too low to move the policy; `reinforce_bucket_ppo_v2.yaml` now defaults to **`1e-3 → 1e-4`**.

## 2026-05-24 PPO Metrics And LR Experiments

### New training metrics

Added to `rl_training/src/ppo.py` and printed from `rl_training/src/train.py` on each log step:

| Metric | Definition | Use |
| --- | --- | --- |
| `explained_variance` | `1 - Var(returns - values) / Var(returns)` | Critic fit quality (R²-style). |
| `approx_kl` | `mean((ratio - 1) - log_ratio)` | How much the policy moved vs rollout policy. |
| `clip_fraction` | `mean(|ratio - 1| > clip_coef)` | How often PPO clipping activates. |

### Experiment setup (local CPU)

All runs used:

- scratch PPO (`--no-bc-init`, no BC checkpoint)
- opponent: score-700 heuristic (`current_heuristic`)
- config base: `reinforce_bucket_ppo_v2.yaml` scaled to **100 updates**, `log_every: 10`
- env: 4 envs × 16 rollout steps, `episode_steps: 200`, fast env
- other PPO hyperparams unchanged: `clip_coef=0.2`, `ent_coef=0.01`, `vf_coef=0.5`, `epochs=3`

### Run A — BC-init smoke vs random (100 updates)

- config: `smoke_ppo_v2.yaml` (100 updates), init from `artifacts/bc/bc_best.pt`
- runtime: ~48s
- checkpoint: `artifacts/rl_current_heuristic_fast/train_100_steps/`
- only **2 finished episodes** in 100 updates (most rollouts truncated mid-game)
- `approx_kl ≈ 0`, `clip_fraction = 0` throughout — policy barely moved from BC
- `explained_variance` ended at **0.44** (noisy batch-to-batch)

### Run B — scratch vs heuristic, old LR (`3e-4 → 3e-5`)

- checkpoint: `artifacts/rl_current_heuristic_fast/scratch_ppo_100_vs_heuristic/`
- runtime: ~2.2 min
- when games finished: `episode_return_mean ≈ -0.99` to **-1.02** (losses)
- `approx_kl` peaked at **0.000009**; `clip_fraction = 0` all steps
- `policy_loss ≈ 0` after update 20 — effectively no learning signal
- final: `explained_variance ≈ 0.26`, `entropy ≈ 4.45`

### Run C — LR sweep, scratch vs heuristic (100 updates each)

| Run | LR schedule | max `approx_kl` | max `clip_fraction` | max `|policy_loss|` | final `entropy` | final `explained_var` | finished-game return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | `3e-4 → 3e-5` | 0.000009 | 0.000 | 0.040 | 4.45 | 0.26 | **-0.99** |
| C2 | `1e-3 → 1e-4` | 0.000326 | 0.000 | 0.029 | 4.13 | 0.32 | **-0.99** |
| C3 | `3e-3 → 3e-4` | **0.006837** | **0.078** | **0.087** | **3.65** | 0.33 | **-1.00** |

Checkpoints:

- `artifacts/rl_current_heuristic_fast/scratch_ppo_100_heuristic_lr3e4/`
- `artifacts/rl_current_heuristic_fast/scratch_ppo_100_heuristic_lr1e3/`
- `artifacts/rl_current_heuristic_fast/scratch_ppo_100_heuristic_lr3e3/`

Notable C3 (3e-3) log points:

- update 10: `approx_kl=0.0068`, `clip_fraction=0.078`, `entropy=4.17`
- update 80: `approx_kl=0.0039`, `clip_fraction=0.027`, `policy_loss=0.087`, `entropy=3.93`

### Conclusions from LR sweep

1. **Old default `3e-4` is too low** for scratch PPO in this env — KL and clip fraction stay at zero; the policy does not move meaningfully in 100 updates.
2. **`1e-3`** produces visible early policy movement without clipping; reasonable new default for scratch PPO.
3. **`3e-3`** finally activates PPO clipping and larger policy updates, but still **no win-rate improvement** vs heuristic in 100 updates; entropy drops faster (more policy change, possibly noisier).
4. None of the 100-update scratch runs beat the heuristic on finished episodes; longer runs or BC init remain necessary.

### Config change

`rl_training/reinforce_bucket_ppo_v2.yaml`:

```yaml
ppo:
  lr: 0.001      # was 0.0003
  lr_end: 0.0001 # was 0.00003
```

BC fine-tune config (`reinforce_bucket_bc_finetune_v2.yaml`) unchanged at `1e-4 → 1e-5`.

Scratch PPO vs heuristic with new default LR:

```bash
cd rl_training
conda run -n ml python -m src.train \
  --config reinforce_bucket_ppo_v2.yaml \
  --no-bc-init
```


Five changes were made in one batch:

### 1. Fixed terminal reward (CRITICAL)

The old `terminal_reward()` returned 0.0 whenever both players had positive ship counts, which is nearly every game. The entire PPO loop was training on zero reward signal.

Fixed: reward is now `+1` (win), `-1` (loss), or `0` (draw) based on relative final score.

### 2. Fixed BC pipeline and added bucket supervision

- Removed dead `fixed_ship_count` import that crashed both `build_bc_dataset.py` and `build_bc_dataset_parquet.py`.
- Added `match_ships_to_bucket()` that maps actual replay `n_ships` to the closest valid ship bucket index.
- Both dataset builders now emit `ship_bucket_index` labels in the `.npz` output.
- `train_bc.py` now trains both `target_logits` (cross-entropy) and `ship_bucket_logits` (cross-entropy on the selected target's bucket row). Logs separate target accuracy and bucket accuracy.

### 3. Added shaped rewards and GAE

- `OrbitWarsEnv` now tracks per-step production, total ships, and planet count. Intermediate steps emit clamped shaping rewards: `delta_production * 0.02 + delta_ships * 0.001 + delta_planets * 0.05`, clamped to `[-0.1, 0.1]`.
- Terminal reward (+1/-1/0) still dominates.
- `PPOConfig` gained `gae_lambda` (default 0.95). Advantage computation in `collect_rollout` switched from MC returns to GAE(lambda).
- `PPOConfig` gained `lr_end` for linear LR annealing.

### 4. Scaled PPO configs

- Created `reinforce_bucket_ppo_v2.yaml` (16 rollout, 4 envs, 200 updates, GAE, LR annealing).
- Created `reinforce_bucket_bc_finetune_v2.yaml` (same scale, lower LR/entropy for BC fine-tune).
- Created `smoke_ppo_v2.yaml` for quick pipeline validation.

### 5. Added per-bucket features to the bucket head

- `features.py` gained `build_bucket_features()` producing a `[candidate_count, ship_bucket_count, 4]` tensor per decision. Each bucket gets: normalized ship count, fraction of surplus, fraction of mission base, and validity flag.
- `TurnBatch` carries `bucket_features` alongside the existing arrays.
- `PlanetPolicy.ship_bucket_head` changed from `joint -> [B,C,S]` to `cat(joint, bucket_feat) -> [B,C,S,1]` per bucket. The bucket head now sees what each bucket actually means.
- All call sites updated: `train.py`, `ppo.py`, `opponents.py`, `eval_vs_current_heuristic.py`, `eval_vs_sniper.py`, `play_vs_sniper.py`, `train_bc.py`.

### Post-overhaul smoke results

Smoke test (50 updates vs random, episode_steps=100): runs in ~22s, 2 episodes completed with positive shaped reward. Pipeline fully functional.

Scratch PPO vs heuristic (200 updates, 16×4, episode_steps=200): completed in ~3.3 min. Policy loses all completed games against heuristic (expected for scratch PPO). Eval vs random: 5/10 (coin-flip, as expected with minimal training).

Old pipeline comparison (pre-fix): scratch PPO got 0/20 vs heuristic and 13/20 vs random, but was training on zero reward signal the entire time.

## Current RL Project Structure

Important files:

| Path | Purpose |
| --- | --- |
| `rl_training/src/features.py` | Parses observations into source rows, candidate target rows, masks, ship buckets, bucket features, and action contexts. |
| `rl_training/src/policy.py` | Candidate-ranking policy with target logits, bucket-feature-aware ship-bucket logits, and value head. |
| `rl_training/src/ppo.py` | PPO sampling, joint target/bucket log-probs, GAE-compatible PPO update. Logs explained variance, approx KL, clip fraction. |
| `rl_training/src/train.py` | Rollout collection with GAE, LR annealing, reward assignment, checkpoint saving, training loop. |
| `rl_training/src/env.py` | Kaggle/fast-env wrapper with win/loss terminal reward and per-step shaped rewards. |
| `rl_training/src/config.py` | Config dataclasses including `gae_lambda` and `lr_end`. |
| `rl_training/src/opponents.py` | Random, current heuristic, and self-play opponents (updated for bucket features). |
| `rl_training/build_bc_dataset.py` | JSON replay BC dataset builder. Now produces `target_index` and `ship_bucket_index` labels. |
| `rl_training/build_bc_dataset_parquet.py` | Parquet replay BC dataset builder. Same label format. |
| `rl_training/train_bc.py` | BC trainer. Trains both target and ship-bucket heads, logs separate accuracies. |
| `rl_training/eval_policy.py` | Generic direct-runner eval versus `heuristic`, `sniper`, or `random`. |

Recent run configs/checkpoints:

| Path | Notes |
| --- | --- |
| `rl_training/reinforce_bucket_ppo.yaml` | Old scratch PPO config. Tiny budget (6400 env steps). |
| `rl_training/reinforce_bucket_ppo_v2.yaml` | Scratch PPO config with GAE, shaped rewards, LR annealing. Default LR **1e-3 → 1e-4** (2026-05-24). |
| `rl_training/reinforce_bucket_bc_finetune_v2.yaml` | BC-initialized PPO fine-tune config (`1e-4 → 1e-5`). |
| `rl_training/smoke_ppo_v2.yaml` | Fast smoke test config (vs random, 50 updates). |
| `rl_training/artifacts/rl_current_heuristic_fast/reinforce_bucket_ppo_v2_vs_heuristic/` | Post-overhaul scratch PPO checkpoints. Baseline only. |
| `rl_training/artifacts/rl_current_heuristic_fast/scratch_ppo_100_heuristic_lr{3e4,1e3,3e3}/` | 2026-05-24 LR sweep checkpoints (100 updates, scratch vs heuristic). |

## Remaining Known Issues

### 1. Credit assignment is still per-step, not per-planet

All source-planet decisions from the same turn receive the same GAE advantage. A good action and a bad action in the same turn are reinforced or punished together. This is a fundamental limitation of the current multi-actor-per-step design.

### 2. BC labels match by angle only

The BC builder labels actions by nearest candidate angle. This is fragile for orbiting planets, friendly staging, and targets with similar bearing. A distance-based or heuristic-planner-based matching would be more robust.

### 3. Candidate selection is still naive

`build_candidates()` uses nearest enemies/neutrals/friendlies. It does not include best-ROI targets, weakest enemies, or heuristic-recommended targets. The best action may be missing from the action set.

### 4. Per-step shaping coefficients are not tuned

The shaping weights (0.02 for production, 0.001 for ships, 0.05 for planets) were chosen as reasonable defaults but have not been validated. They may over- or under-weight different aspects of play.

### 5. Training against heuristic is slow

Each env step against the heuristic opponent takes ~0.5s. A 200-update run (16×4 rollout) takes ~3 min. Longer runs of 1000+ updates need 15+ min. Consider training vs random or weaker opponents for faster iteration.

## Recommended Next Steps

1. **Generate fresh BC dataset** with bucket labels using the fixed `build_bc_dataset.py`.
2. **Train BC** on both target and bucket heads; verify target accuracy >70%, bucket accuracy >40%.
3. **Evaluate BC checkpoint** vs random — should win >15/20.
4. **PPO fine-tune from BC** using `reinforce_bucket_bc_finetune_v2.yaml`.
5. **Evaluate PPO** vs heuristic — target: >5/20 as first milestone.
6. If results are promising, increase training budget or add richer features (incoming fleets, ROI, travel time).

## Useful Commands

Smoke test the pipeline (fast, vs random):

```bash
cd rl_training
conda run -n ml python -m src.train --config smoke_ppo_v2.yaml
```

Scratch PPO vs heuristic (~3 min for 200 updates):

```bash
cd rl_training
conda run -n ml python -m src.train --config reinforce_bucket_ppo_v2.yaml
```

BC-initialized PPO fine-tune (run after training BC):

```bash
cd rl_training
conda run -n ml python -m src.train \
  --config reinforce_bucket_bc_finetune_v2.yaml \
  --checkpoint artifacts/bc/bc_policy.pt \
  --reset-optimizer
```

Train behavior cloning (after generating a dataset with bucket labels):

```bash
cd rl_training
conda run -n ml python train_bc.py \
  --config default_cfg.yaml \
  --dataset artifacts/bc/top_players_bc.npz \
  --output artifacts/bc/bc_policy.pt \
  --epochs 10 \
  --batch-size 2048
```

Evaluate a checkpoint vs random/heuristic/sniper:

```bash
cd rl_training
conda run -n ml python eval_policy.py \
  --config reinforce_bucket_ppo_v2.yaml \
  --checkpoint artifacts/rl_current_heuristic_fast/reinforce_bucket_ppo_v2_vs_heuristic/ckpt_last.pt \
  --baseline random \
  --games 20 \
  --seed-start 5000 \
  --episode-steps 200 \
  --device cpu \
  --deterministic
```
