# Kaggle GPU training & submission guide

End-to-end workflow for training the JAX Transformer PPO policy on Kaggle's
free GPU and uploading the resulting bot. The training code already lives in
`rl_training_jax/`; this guide just tells you how to wire it into Kaggle.

```
┌──────────────────┐  upload repo   ┌─────────────────────────┐  train +    ┌──────────────────┐
│  this repo (PC)  │ ─────────────► │  Kaggle GPU notebook    │ ──────────► │ submission_jax   │
│                  │  as dataset    │  (kaggle/kaggle_train)  │  export     │ .zip in /output  │
└──────────────────┘                └─────────────────────────┘             └──────────────────┘
                                                                                   │
                                                                                   ▼
                                                                       upload to Orbit Wars
                                                                       competition page
```

## 0. Prep: bundle the repo into a Kaggle Dataset

Kaggle GPU notebooks can read from datasets but cannot `git clone` random
repos efficiently. The cleanest path is to package this repository as a
Kaggle Dataset.

1. Locally, zip the repo (excluding heavy/non-essential dirs):

   ```bash
   cd ..
   zip -r orbit-wars-rl-template.zip orbit_wars_cursor_template_v2 \
     -x "*/.git/*" "*/artifacts/*" "*/__pycache__/*" "*.pyc" \
        "*/notebooks/*" "*/analysis/*"
   ```

2. Go to <https://www.kaggle.com/datasets> → **New Dataset** → upload the zip.
   Title it `orbit-wars-rl-template` (or any name — note the slug Kaggle
   assigns).

3. (Optional) Add a second dataset entry as a "private dataset" so updates
   stay private.

When you attach the dataset to a notebook, files appear at
`/kaggle/input/<dataset-slug>/`.

## 1. Create the training notebook

1. Kaggle home → **+ Create** → **New Notebook**.
2. Right-hand sidebar:
   - **Accelerator**: `GPU T4 x2` (or `P100` — either works; T4 x2 is fine, we
     only use one device for now).
   - **Internet**: ON. We only need it for the first cell to install JAX-CUDA.
3. Click **Add Data** → search for your dataset slug → **Add**.
4. In the notebook, paste the contents of
   `kaggle/kaggle_train.py` (this repo). Split it into one cell per
   `# ---- Cell N: ...` marker so you can step through them.
5. If your dataset slug differs from `orbit-wars-rl-template`, change
   `DATASET_NAME` at the top of cell 1.

## 2. Run the cells

| Cell | Time | Purpose |
|---|---|---|
| 1 — environment | ~30–90 s | Copy repo into `/kaggle/working/repo`, install `jax[cuda12_pip]` (only if Kaggle's preinstalled `jax` is CPU-only). |
| 2 — smoke run | ~2 min on GPU | Runs `configs/smoke_transformer.yaml` (20 updates). Look for finite losses, KL≈0 in later updates, and `env_sps` > 100. |
| 3 — real training | hours, depending on `total_updates` | Runs `configs/transformer_selfplay.yaml`. Tail the log; you should see episodes complete and `mean_ret` drift around 0 in self-play. |
| 4 — export | ~5 s | Calls `scripts/export_jax_submission.py` and writes `submission_jax.zip` to `/kaggle/working/`. |

You can stop cell 3 at any time (the trainer checkpoints every 100 updates).
Cell 4 picks up the latest checkpoint.

## 3. Tuning `transformer_selfplay.yaml` on Kaggle

The defaults are sized for a Kaggle session (~9 h wall-clock budget):

```yaml
env:
  num_envs: 64          # raise if you have GPU RAM to spare
  episode_steps: 500    # full Orbit Wars episode
  rollout_steps: 32

model:
  d_model: 96
  num_heads: 4
  num_layers: 3
  bucket_count: 8

ppo:
  total_updates: 5000   # ~3-6 hours on T4 with these settings
  epochs: 3
  minibatch_size: 2048
  lr_start: 1.0e-3
  lr_end: 1.0e-5
```

Tips:

- **Watch GPU RAM**: each env contributes ~MAX_PLANETS × MAX_FLEETS × d_model
  to attention memory. If you OOM, halve `num_envs` first.
- **Don't bump d_model past 128** unless you also raise `total_updates`. A
  bigger model trained briefly is usually worse than a smaller one trained
  longer.
- **Save more often**: drop `checkpoint_every` to 50 when iterating so a
  crashed cell still leaves you a usable bot.
- **Restart notebook**: long Kaggle sessions sometimes get killed. The
  trainer auto-resumes if you change the run name and copy the previous
  `ckpt_last.npz` into the new run dir — or just rerun from scratch.

## 4. What the training log tells you

Each row prints:

```
update | upd/s | env_sps | rollout_s | train_s | episodes | mean_ret | loss | policy | value | entropy | ev | approx_kl | clip_frac
```

Healthy signs (after the first ~50 updates):

- `mean_ret` oscillates around 0 (self-play balance). If it drifts strongly
  positive or negative, the learner is overfitting to one player's
  perspective — check `learner_players_np` distribution.
- `ev` (explained variance) climbs above 0.5 and stays there — the critic is
  fitting the returns.
- `entropy` slowly decays from ~4.5 toward ~2.0 over thousands of updates.
  A sudden crash to near-0 means the policy collapsed; bump `ent_coef`.
- `approx_kl` stays under ~0.02 most of the time. Spikes above 0.05 indicate
  the LR is too high.
- `clip_fraction` between 0.05–0.3 — too low = nothing to clip = LR too low,
  too high = LR too high.

## 5. Download the submission zip

In the notebook UI, right-hand panel → **Output** tab → click
`submission_jax.zip` → **Download**.

The zip is small (~300–800 KiB; mostly weights). Unpack it locally to
sanity-check the structure:

```
submission_jax/
├── main.py                # Kaggle entry (imports agent)
├── src/
│   ├── jax_bot.py         # inference loop
│   ├── policy.py          # Flax model definition
│   └── orbit_wars/        # minimal subset: constants/state/geometry/convert/features_jax/decode
└── weights/
    ├── policy.msgpack     # flax-serialized params
    └── model_config.json  # d_model/n_heads/n_layers/feature dims
```

## 6. Validate the bundle locally before submitting

```bash
conda run -n ml python kaggle/test_submission_locally.py \
    --submission submission_jax.zip
```

Expected output:

```
first call (JIT compile): ~9000 ms
  moves: [[...], [...]]
steady-state: ~25 ms / call
empty board OK
SUBMISSION OK.
```

If you see a non-zero exit code, the bundle is broken — re-run cell 4 on
Kaggle.

## 7. Submit to the competition

1. Go to the Orbit Wars competition page on Kaggle → **Submit**.
2. Choose **Upload Submission** → select `submission_jax.zip`.
3. Wait for the validation tournament to score it.

Per-call latency of ~25 ms (after the one-time ~10 s JIT warm-up at game
start) is comfortably under Kaggle's per-step budget.

## 8. Iterating

Typical flow after the first submission:

1. Edit configs or `rl_training_jax/src/` locally.
2. **Update the Kaggle dataset**: open your dataset → **New Version** →
   upload the new zip → **Save**.
3. In the notebook: **File → Add or Update Dataset** → confirm. Cell 1 will
   re-copy the new tree.
4. Re-run cells 2 → 4.

For evaluation locally against `kaggle700_current_heuristic`, see Phase 8 (to
be added: `scripts/eval_jax_policy.py`).

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `jax.devices() == [CpuDevice]` | GPU not attached or jax[cuda] not installed | Re-run cell 1; verify Accelerator = GPU. |
| OOM during training | `num_envs` too large | Halve `num_envs` in YAML. |
| First call after restart hangs ~30 s | JAX recompiling kernels | Normal. Subsequent calls are fast. |
| Loss = NaN | LR too high or fp32 underflow in masked log_softmax | Drop `lr_start` to 5e-4. |
| `mean_ret` stuck at exactly 0.000 | No episodes completing within `rollout_steps × total_updates` window | Raise `total_updates` or lower `episode_steps`. |
| Submission returns `[]` for every step | Weight load failed silently | Set `ORBIT_WARS_DEBUG=1` env var and inspect Kaggle logs. |

## Quick reference

```bash
# Smoke locally before sending to Kaggle
cd rl_training_jax
conda run -n ml python -m train_ppo --config configs/smoke_transformer.yaml

# Export a zip from a local checkpoint
conda run -n ml python scripts/export_jax_submission.py \
    --checkpoint artifacts/jax_smoke_transformer/ckpt_last.npz \
    --config configs/smoke_transformer.yaml \
    --output ../submission_jax.zip

# Validate the zip
cd ..
conda run -n ml python kaggle/test_submission_locally.py --submission submission_jax.zip
```
