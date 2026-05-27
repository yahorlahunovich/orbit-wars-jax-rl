#!/usr/bin/env python3
"""Compare env + policy steps/sec: PyTorch rl_training vs JAX rl_training_jax."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rl_training_jax" / "src"))
sys.path.insert(0, str(REPO / "rl_training"))

from orbit_wars.reset import reset
from orbit_wars.step import step, step_jit
from orbit_wars.step import _list_action_to_padded


def bench_jax_env(steps: int, warmup: int, seed: int) -> float:
    state = reset(seed, episode_steps=steps + 5)
    empty_a, empty_m = _list_action_to_padded([])
    for _ in range(warmup):
        state = step(state, [[] , []])
        if bool(state.done):
            state = reset(seed, episode_steps=steps + 5)
    state = reset(seed, episode_steps=steps + 5)
    t0 = time.perf_counter()
    for _ in range(steps):
        state = step(state, [[] , []])
        if bool(state.done):
            break
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def bench_jax_env_jit_only(steps: int, warmup: int, seed: int) -> float:
    state = reset(seed, episode_steps=steps + 5)
    a0, m0 = _list_action_to_padded([])
    a1, m1 = _list_action_to_padded([])
    step_jit(state, a0, a1, m0, m1)  # compile
    for _ in range(warmup):
        state = step_jit(state, a0, a1, m0, m1)
    state.planets.block_until_ready()
    state = reset(seed, episode_steps=steps + 5)
    t0 = time.perf_counter()
    for _ in range(steps):
        state = step_jit(state, a0, a1, m0, m1)
    state.planets.block_until_ready()
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def bench_jax_env_vmap(batch: int, steps: int, warmup: int, seed: int) -> float:
    import jax.tree_util as tu

    from orbit_wars.step import batched_step

    states = [reset(seed + i, episode_steps=steps + 5) for i in range(batch)]
    batched = tu.tree_map(lambda *xs: jnp.stack(xs), *states)
    a, m = _list_action_to_padded([])
    a_b = jnp.broadcast_to(a, (batch, *a.shape))
    m_b = jnp.broadcast_to(m, (batch, *m.shape))
    for _ in range(warmup):
        batched = batched_step(batched, a_b, a_b, m_b, m_b)
    batched.planets.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(steps):
        batched = batched_step(batched, a_b, a_b, m_b, m_b)
    batched.planets.block_until_ready()
    elapsed = time.perf_counter() - t0
    return (steps * batch) / elapsed


def bench_jax_policy(batch: int, steps: int, warmup: int) -> float:
    from policy import PlanetPolicy

    rng = jax.random.PRNGKey(0)
    model = PlanetPolicy(candidate_count=49, ship_bucket_count=5, global_dim=16)
    example = {
        "self_features": jnp.zeros((batch, 22), dtype=jnp.float32),
        "candidate_features": jnp.zeros((batch, 49, 28), dtype=jnp.float32),
        "global_features": jnp.zeros((batch, 16), dtype=jnp.float32),
        "candidate_mask": jnp.ones((batch, 49), dtype=jnp.bool_),
        "ship_bucket_mask": jnp.ones((batch, 49, 5), dtype=jnp.bool_),
        "bucket_features": jnp.zeros((batch, 49, 5, 4), dtype=jnp.float32),
    }
    params = model.init(rng, **example)
    apply = jax.jit(lambda p, x: model.apply(p, **x))

    for _ in range(warmup):
        apply(params, example)

    t0 = time.perf_counter()
    for _ in range(steps):
        apply(params, example)
    elapsed = time.perf_counter() - t0
    return (steps * batch) / elapsed


def bench_torch_policy(batch: int, steps: int, warmup: int) -> float:
    import torch
    from src.policy import PlanetPolicy as TorchPolicy

    device = torch.device("cpu")
    model = TorchPolicy(
        self_dim=22,
        candidate_dim=28,
        global_dim=16,
        candidate_count=49,
        ship_bucket_count=5,
    ).to(device)
    model.eval()
    example = (
        torch.zeros(batch, 22, device=device),
        torch.zeros(batch, 49, 28, device=device),
        torch.zeros(batch, 16, device=device),
        torch.ones(batch, 49, dtype=torch.bool, device=device),
        torch.ones(batch, 49, 5, dtype=torch.bool, device=device),
        torch.zeros(batch, 49, 5, 4, device=device),
    )
    with torch.inference_mode():
        for _ in range(warmup):
            model(*example)
        t0 = time.perf_counter()
        for _ in range(steps):
            model(*example)
    elapsed = time.perf_counter() - t0
    return (steps * batch) / elapsed


def bench_torch_env(steps: int, warmup: int, seed: int) -> float:
    import torch
    from src.config import load_train_config
    from src.env import OrbitWarsEnv
    from src.opponents import KaggleRandomOpponent

    cfg = load_train_config(REPO / "rl_training" / "configs" / "smoke_ppo.yaml")
    opp = KaggleRandomOpponent()
    env = OrbitWarsEnv(cfg, opp)
    env.reset(seed=seed)
    for _ in range(warmup):
        env.step([])
        if env.last_obs is None:
            env.reset(seed=seed)
    env.reset(seed=seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step([])
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs JAX Orbit Wars training stack")
    parser.add_argument("--env-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print()

    jax_env_sps = bench_jax_env(args.env_steps, args.warmup, args.seed)
    jax_jit_sps = bench_jax_env_jit_only(args.env_steps, args.warmup, args.seed)
    torch_env_sps = bench_torch_env(args.env_steps, args.warmup, args.seed)
    jax_pol_sps = bench_jax_policy(args.batch, args.policy_steps, args.warmup)
    torch_pol_sps = bench_torch_policy(args.batch, args.policy_steps, args.warmup)

    print("=== Environment step throughput (noop vs random, includes feature encoding for torch) ===")
    print(f"  PyTorch rl_training env:     {torch_env_sps:8.1f} steps/s")
    print(f"  JAX env (full step):         {jax_env_sps:8.1f} steps/s")
    print(f"  JAX env (JIT physics only):  {jax_jit_sps:8.1f} steps/s")
    print(f"  Env speedup (full JAX/PyTorch): {jax_env_sps / max(torch_env_sps, 1e-6):.2f}x")
    print()
    print("=== Batched JAX env throughput (vmap over reset seeds, pure JIT) ===")
    for b in (4, 16, 32, 64):
        sps = bench_jax_env_vmap(b, max(args.env_steps // 2, 50), args.warmup, args.seed)
        print(f"  vmap({b:3d}) batched_step:        {sps:8.1f} env-steps/s   ({sps / b:7.1f}/env)")
    print()
    print(f"=== Policy forward pass (batch={args.batch}) ===")
    print(f"  PyTorch PlanetPolicy CPU:    {torch_pol_sps:8.1f} rows/s")
    print(f"  JAX Flax PlanetPolicy CPU:   {jax_pol_sps:8.1f} rows/s")
    print(f"  Policy speedup (JAX/PyTorch):  {jax_pol_sps / max(torch_pol_sps, 1e-6):.2f}x")
    print()
    print("Note: On Kaggle GPU, rerun with JAX_CUDA visible — expect much higher JAX policy SPS.")
    print("      Env JIT core can be vmapped for num_envs parallel rollouts on GPU.")


if __name__ == "__main__":
    main()
