# Orbit Wars Pure JAX PPO Pipeline

This repository contains the pure JAX PPO reinforcement learning pipeline for the Kaggle Orbit Wars environment. It was built by adapting the highly optimized [PureJAXRL](https://github.com/luchris429/purejaxrl) framework to work with a custom Transformer policy and a jittable wrapper around the Orbit Wars engine.

## Architecture Overview

### 1. Environment Wrapper (`src/env.py`)
- **`OrbitWarsPureJaxEnv`**: A wrapper that makes the Python/Kaggle-based Orbit Wars environment compatible with `jax.lax.scan`.
- Uses `jax.pure_callback(..., vmap_method="sequential")` to step out of the compiled XLA graph to handle Kaggle environment resets and the NumPy/SciPy-based comet spawning logic.
- Returns fully padded `OrbitWarsState` and feature dictionaries (planet features & masks) compatible with `vmap`.

### 2. Policy Network (`src/policy.py`)
- **`PlanetPolicy`**: A Transformer-based architecture implemented in Flax.
- It consumes planet features (`P=40` max planets) and outputs:
  1. `target_logits`: Which planet to target (shape `P x P`).
  2. `bucket_logits`: How many ships to send, discretized into buckets (e.g., 25%, 50%, 75%, 100%).
  3. `value`: The baseline critic value.

### 3. PPO Loop (`src/ppo.py`)
- The core algorithm is located in `make_train(config)`.
- It uses a nested `jax.lax.scan` architecture to run episodes (`NUM_STEPS`), compute Generalized Advantage Estimation (GAE), and then update the network across mini-batches and epochs.
- It includes complex custom masked log-probability and entropy calculations (Target Phase + Bucket Phase) for the two-tiered action space.
- Tracks metrics like `approx_kl`, `clip_frac`, and `explained_variance`.

### 4. Rollout and Decoding (`src/orbit_wars/rollout.py`, `src/orbit_wars/decode.py`)
- Uses a split-phase grid composition to avoid `O(P*P*B)` memory bottlenecks.
- Phase 1: Calculates valid targets.
- Phase 2: Given a sampled target, computes specific physical intercept mechanics to determine bucket validity.

## Future Agent Instructions
- **Do NOT remove the `jax.pure_callback` calls in `env.py`** unless you have fully rewritten the Kaggle comet spawning logic (which relies on `scipy.spatial.distance.cdist`) into pure JAX.
- The initial JIT compilation of `make_train` takes a significant amount of time (often several minutes) because the entire rollout and backpropagation loop is unrolled into a single XLA graph. This is normal.
- When making modifications to the PPO loss or action spaces, ensure that you respect the boolean masks (`target_has_bucket`, `chosen_bucket_valid`, `executed_mask`) as invalid paths must be masked to `-inf` before softmax to prevent `NaN` gradients.
- To run training: `python train.py --config default_cfg.yaml`
- To run evaluations against heuristics: `python eval_vs_sniper.py` or `python play_vs_sniper.py`.
