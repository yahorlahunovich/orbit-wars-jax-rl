# JAX RL Training for Orbit Wars

This package (`rl_training_jax/`) is a **JAX-native** rewrite of the PyTorch stack in `rl_training/`:

| Component | PyTorch (`rl_training/`) | JAX (`rl_training_jax/`) |
|-----------|------------------------|--------------------------|
| Environment | Kaggle `env.run` + Python interpreter | Padded `OrbitWarsState` + `@jit step_jit` |
| Policy | `torch.nn` PlanetPolicy | Flax `PlanetPolicy` |
| Optimizer | Adam (PyTorch) | Optax |
| Validation | smoke PPO + eval scripts | `pytest` parity vs official env |

## Why JAX?

1. **Batched simulation on GPU** — `jax.vmap(step_jit)` runs hundreds of envs in parallel.
2. **End-to-end on device** — policy + env step without CPU↔GPU copies.
3. **Kaggle GPU** — T4/P100 works with `jax[cuda12]` (see below).

## Quick start (local CPU)

```bash
cd rl_training_jax
pip install -e ".[dev]"  # or: pip install jax flax optax pytest pyyaml

# Run tests (parity vs official Kaggle env)
PYTHONPATH=src:../rl_training python -m pytest tests/ -v

# Speed comparison vs PyTorch
PYTHONPATH=src:../rl_training python scripts/bench_speed.py --env-steps 200 --batch 64
```

## Architecture

```
rl_training_jax/src/
  orbit_wars/
    constants.py   # game constants + padded array limits
    geometry.py    # JAX collision math (tested vs reference)
    state.py       # flax.struct OrbitWarsState
    convert.py     # Python obs <-> padded arrays
    reset.py       # init via reference env (exact parity)
    step.py        # JIT physics + Python comet spawn
    reference.py   # bridge to kaggle_environments
  policy.py        # Flax PlanetPolicy
  ppo.py           # PPO loss helpers
```

**Reset** uses the official Kaggle env once (deterministic, seed-safe), then packs state into padded JAX arrays.

**Step** runs comet expiry/spawn in Python (rare, reference-compatible), then `@jit step_jit` for production, fleet movement, combat, termination.

**Tests** roll out noop games and compare planet/fleet tuples against the reference env (`tests/test_parity.py`).

## Kaggle GPU upload

Kaggle notebooks cannot pip-install arbitrary packages at runtime in all cases, but **JAX + Flax + Optax are supported** on GPU notebooks when you add them as dataset dependencies.

### Option A — Add as Kaggle dataset (recommended)

1. On your machine, build a wheel bundle:

```bash
pip download jax[cuda12] flax optax chex -d kaggle_jax_wheels
zip -r orbit_wars_jax_deps.zip kaggle_jax_wheels
```

2. Upload `orbit_wars_jax_deps.zip` as a Kaggle dataset.
3. In your training notebook:

```python
!pip install -q /kaggle/input/orbit-wars-jax-deps/*

import sys
sys.path.insert(0, "/kaggle/working/orbit_wars/rl_training_jax/src")
sys.path.insert(0, "/kaggle/working/orbit_wars/rl_training")

# Verify GPU
import jax
print(jax.devices())  # should show CudaDevice
```

4. Copy the repo (or subset) into `/kaggle/working/orbit_wars/`:

```python
# If using git:
!git clone https://github.com/YOUR_USER/orbit-wars.git /kaggle/working/orbit_wars
```

Or add this repo as a second Kaggle dataset.

### Option B — Bundle env in notebook

The JAX env has **no dependency on kaggle_environments at training time** (only for parity tests). For Kaggle training you only need:

- `rl_training_jax/src/orbit_wars/` (full JAX env)
- `rl_training/src/features.py` + dependencies (feature encoding, reused)
- `rl_training_jax/src/policy.py`, `ppo.py`

You do **not** need `analysis/fast_kaggle_env/` on Kaggle for JAX training.

### Training command on Kaggle GPU

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

from orbit_wars.reset import reset
from orbit_wars.step import step

state = reset(seed=42, episode_steps=500)
for _ in range(1000):
    state = step(state, [[], []])
```

For PPO, use `jax.vmap` over reset seeds and JIT-compiled rollouts (see `scripts/bench_speed.py`).

### Submission bot (separate from training)

Kaggle **submission** still requires `main.py` + heuristic/RL bot under `versions/`. Export JAX policy weights to NumPy/Torch for inference, or write a lightweight JAX CPU inference `main.py` (JAX CPU wheels are small).

## Validation checklist

Before trusting JAX training numbers:

1. `pytest tests/test_parity.py` — state matches reference env (noop rollouts).
2. `pytest tests/test_geometry.py` — collision math matches reference.
3. `python scripts/bench_speed.py` — confirm env/policy speedup locally.
4. On Kaggle GPU: rerun bench with `--batch 256` after JIT warmup.

## Known limitations (v0.1)

- Comet spawn runs in Python (5× per episode); negligible vs fleet physics.
- Feature encoding still reuses NumPy code from `rl_training/src/features.py` (next: JAX `encode_turn`).
- Full PPO training loop is scaffolded (`ppo.py`); wire into `train.py` for production runs.
- Batched `vmap(step_jit)` skips Python comet path — use episodes < 50 steps for pure-vmapped benchmarks, or pre-spawn comets.

## Next steps

1. Port `encode_turn` to JAX for zero-copy rollouts.
2. Implement `train.py` with `vmap` env + GAE on GPU.
3. Export checkpoint → Kaggle `main.py` inference wrapper.
