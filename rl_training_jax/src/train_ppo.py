"""PPO trainer for the JAX Transformer Orbit Wars policy.

End-to-end design:

- `num_envs` envs are stacked via `tree_map(jnp.stack)` and stepped with
  `jax.vmap(step_jit)` inside a `jax.lax.scan` of length `rollout_steps`.
- Self-play: both players use the same `params`. Per env we randomly assign
  which player is the learner at reset time; PPO trains on the learner's
  decision rows only. Opponent's actions are sampled from the same policy
  (no gradient) and fed to the env.
- Reward: pure terminal +1 / 0 / -1 from `state.rewards[learner_player]`.
  No shaping.
- GAE: gamma = 0.9999, lambda = 0.95 by default.
- Optimizer: optax cosine-decayed LR with `clip_by_global_norm`.

The trainer is structured so the rollout and the PPO update each live in a
single jit. Comet spawning happens host-side only when the env step reaches
one of `COMET_SPAWN_STEPS`; otherwise the entire rollout stays on device.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import optax

from orbit_wars import (
    BUCKET_COUNT,
    COMET_SPAWN_STEPS,
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    batched_step,
    compose_action_grid,
    encode_observation,
    reset,
)
from orbit_wars.rollout import pack_padded_actions, sample_actions
from orbit_wars.state import OrbitWarsState
from orbit_wars.step import _maybe_spawn_comet_numpy
from policy import PlanetPolicy
from ppo import compute_gae, explained_variance, ppo_loss_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainConfig:
    seed: int = 0
    run_name: str = "jax_ppo_transformer"
    save_dir: str = "artifacts"

    # Env / batch
    num_envs: int = 16
    episode_steps: int = 200
    rollout_steps: int = 32

    # Model
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    bucket_count: int = BUCKET_COUNT

    # PPO
    total_updates: int = 200
    epochs: int = 3
    minibatch_size: int = 1024
    gamma: float = 0.9999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr_start: float = 1e-3
    lr_end: float = 1e-5
    max_grad_norm: float = 0.5

    # Logging / checkpoint
    log_every: int = 1
    checkpoint_every: int = 50


def load_config(path: str | Path) -> TrainConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    env = data.get("env", {})
    model = data.get("model", {})
    ppo = data.get("ppo", {})
    return TrainConfig(
        seed=int(data.get("seed", 0)),
        run_name=str(data.get("run_name", "jax_ppo_transformer")),
        save_dir=str(data.get("save_dir", "artifacts")),
        num_envs=int(env.get("num_envs", 16)),
        episode_steps=int(env.get("episode_steps", 200)),
        rollout_steps=int(env.get("rollout_steps", 32)),
        d_model=int(model.get("d_model", 96)),
        num_heads=int(model.get("num_heads", 4)),
        num_layers=int(model.get("num_layers", 3)),
        bucket_count=int(model.get("bucket_count", BUCKET_COUNT)),
        total_updates=int(ppo.get("total_updates", 200)),
        epochs=int(ppo.get("epochs", 3)),
        minibatch_size=int(ppo.get("minibatch_size", 1024)),
        gamma=float(ppo.get("gamma", 0.9999)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_coef=float(ppo.get("clip_coef", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        lr_start=float(ppo.get("lr_start", 1e-3)),
        lr_end=float(ppo.get("lr_end", 1e-5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        log_every=int(data.get("log_every", 1)),
        checkpoint_every=int(data.get("checkpoint_every", 50)),
    )


# ---------------------------------------------------------------------------
# Env management (host-side comet spawn + on-device step)
# ---------------------------------------------------------------------------


def make_initial_states(cfg: TrainConfig, seed_base: int) -> tuple[OrbitWarsState, np.ndarray]:
    """Build initial batched states and per-env learner-player assignment."""
    rng = random.Random(seed_base)
    states = []
    learner_players = np.zeros(cfg.num_envs, dtype=np.int32)
    for i in range(cfg.num_envs):
        states.append(reset(seed_base + i, episode_steps=cfg.episode_steps))
        learner_players[i] = rng.randint(0, 1)
    batched = tu.tree_map(lambda *xs: jnp.stack(xs), *states)
    return batched, learner_players


def maybe_spawn_comets_host(batched_states: OrbitWarsState, cfg: TrainConfig) -> OrbitWarsState:
    """Run host-side comet spawn on each env only when the env's next step is
    a comet-spawn step. Cheap: 0 envs touch most updates."""
    next_step = int(np.asarray(batched_states.step)[0]) + 1
    if next_step not in COMET_SPAWN_STEPS:
        return batched_states
    new_states = []
    n = int(batched_states.step.shape[0])
    for i in range(n):
        single = tu.tree_map(lambda x, i=i: x[i], batched_states)
        new_states.append(_maybe_spawn_comet_numpy(single))
    return tu.tree_map(lambda *xs: jnp.stack(xs), *new_states)


# ---------------------------------------------------------------------------
# Rollout (one rollout-step batched across envs)
# ---------------------------------------------------------------------------


def policy_apply_factory(model: PlanetPolicy):
    def apply_fn(params, **kwargs):
        return model.apply(params, **kwargs)

    return jax.jit(apply_fn)


def rollout_step_factory(model: PlanetPolicy):
    @jax.jit
    def step_one(states: OrbitWarsState, params, rng, learner_players):
        # Player 0 view
        rng, k0, k1 = jax.random.split(rng, 3)
        feats0 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(0))
        out0 = model.apply(params, **feats0)
        grid0 = jax.vmap(compose_action_grid, in_axes=(0, None))(states, jnp.int32(0))
        s0 = sample_actions(k0, out0.target_logits, out0.bucket_logits, grid0)
        a0, m0 = pack_padded_actions(s0["target_idx"], s0["bucket_idx"], s0["source_valid"], grid0)

        # Player 1 view (same params, fresh randomness)
        feats1 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(1))
        out1 = model.apply(params, **feats1)
        grid1 = jax.vmap(compose_action_grid, in_axes=(0, None))(states, jnp.int32(1))
        s1 = sample_actions(k1, out1.target_logits, out1.bucket_logits, grid1)
        a1, m1 = pack_padded_actions(s1["target_idx"], s1["bucket_idx"], s1["source_valid"], grid1)

        new_states = jax.vmap(__import__("orbit_wars.step", fromlist=["step_jit"]).step_jit)(
            states, a0, a1, m0, m1
        )

        # Learner's perspective: gather per-env from (feats0/feats1, out0/out1, grid0/grid1, s0/s1).
        def gather_by_player(zero_t, one_t):
            return jnp.where(
                learner_players.astype(jnp.bool_)[:, None] if zero_t.ndim == 2 else
                learner_players.astype(jnp.bool_)[:, None, None] if zero_t.ndim == 3 else
                learner_players.astype(jnp.bool_)[:, None, None, None],
                one_t, zero_t,
            ) if zero_t.ndim > 1 else jnp.where(learner_players.astype(jnp.bool_), one_t, zero_t)

        # Per-row info we keep for PPO. All are per-env-time shapes (B, ...).
        learner_feats = jax.tree_util.tree_map(gather_by_player, feats0, feats1)
        learner_value = jnp.where(learner_players.astype(jnp.bool_), out1.value, out0.value)
        target_idx = jnp.where(learner_players.astype(jnp.bool_)[:, None], s1["target_idx"], s0["target_idx"])
        bucket_idx = jnp.where(learner_players.astype(jnp.bool_)[:, None], s1["bucket_idx"], s0["bucket_idx"])
        log_prob = jnp.where(learner_players.astype(jnp.bool_)[:, None], s1["log_prob"], s0["log_prob"])
        source_valid = jnp.where(learner_players.astype(jnp.bool_)[:, None], s1["source_valid"], s0["source_valid"])
        target_has_bucket = jnp.where(
            learner_players.astype(jnp.bool_)[:, None, None],
            jnp.any(grid1["bucket_valid"], axis=-1) & grid1["pair_valid"],
            jnp.any(grid0["bucket_valid"], axis=-1) & grid0["pair_valid"],
        )
        bucket_valid = jnp.where(
            learner_players.astype(jnp.bool_)[:, None, None, None],
            grid1["bucket_valid"], grid0["bucket_valid"],
        )

        # Terminal reward seen by the learner.
        reward = jnp.where(
            new_states.done & (learner_players == 0),
            new_states.rewards[:, 0],
            jnp.where(
                new_states.done & (learner_players == 1),
                new_states.rewards[:, 1],
                jnp.zeros_like(new_states.rewards[:, 0]),
            ),
        )
        done = new_states.done

        record = {
            "planet_features": learner_feats["planet_features"],
            "planet_mask": learner_feats["planet_mask"],
            "fleet_features": learner_feats["fleet_features"],
            "fleet_mask": learner_feats["fleet_mask"],
            "global_features": learner_feats["global_features"],
            "target_idx": target_idx,
            "bucket_idx": bucket_idx,
            "log_prob": log_prob,
            "source_valid": source_valid,
            "target_has_bucket": target_has_bucket,
            "bucket_valid": bucket_valid,
            "value": learner_value,
            "reward": reward,
            "done": done,
        }
        return new_states, record, rng

    return step_one


def reset_done_envs(states: OrbitWarsState, dones_np: np.ndarray, next_seed: int, cfg: TrainConfig) -> tuple[OrbitWarsState, int, np.ndarray]:
    """Host-side resets for envs whose episodes ended. Returns (new_states,
    next_seed, new_learner_players_for_those_envs)."""
    n = int(states.step.shape[0])
    if not dones_np.any():
        return states, next_seed, np.zeros(n, dtype=np.int32)

    rng = random.Random(next_seed)
    refreshed = []
    new_lp = np.zeros(n, dtype=np.int32)
    states_list = [tu.tree_map(lambda x, i=i: x[i], states) for i in range(n)]
    for i in range(n):
        if dones_np[i]:
            s = reset(next_seed, episode_steps=cfg.episode_steps)
            new_lp[i] = rng.randint(0, 1)
            next_seed += 1
            refreshed.append(s)
        else:
            refreshed.append(states_list[i])
    new_states = tu.tree_map(lambda *xs: jnp.stack(xs), *refreshed)
    return new_states, next_seed, new_lp


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def make_optimizer(cfg: TrainConfig):
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.lr_start,
        decay_steps=cfg.total_updates,
        alpha=cfg.lr_end / max(cfg.lr_start, 1e-12),
    )
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule)), schedule


def init_policy_params(rng, model: PlanetPolicy):
    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
        "fleet_features": jnp.zeros((1, MAX_FLEETS, FLEET_FEATURE_DIM), jnp.float32),
        "fleet_mask": jnp.ones((1, MAX_FLEETS), jnp.bool_),
        "global_features": jnp.zeros((1, GLOBAL_FEATURE_DIM), jnp.float32),
    }
    return model.init(rng, **example), example


def make_update_step(model: PlanetPolicy, optimizer, cfg: TrainConfig):
    @jax.jit
    def update(params, opt_state, batch):
        def loss(p):
            return ppo_loss_fn(p, model.apply, batch, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef)

        (l, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l, metrics

    return update


def train(cfg: TrainConfig) -> None:
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    model = PlanetPolicy(
        planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS,
        d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        bucket_count=cfg.bucket_count,
    )
    params, _ = init_policy_params(init_rng, model)
    optimizer, _ = make_optimizer(cfg)
    opt_state = optimizer.init(params)

    rollout_step = rollout_step_factory(model)
    update_step = make_update_step(model, optimizer, cfg)

    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_base = cfg.seed * 10000 + 1
    states, learner_players_np = make_initial_states(cfg, seed_base)
    learner_players = jnp.asarray(learner_players_np)
    next_seed = seed_base + cfg.num_envs

    print(
        f"JAX devices: {jax.devices()} | envs={cfg.num_envs} rollout={cfg.rollout_steps} "
        f"updates={cfg.total_updates}"
    )
    print(
        "update | upd/s | env_sps | rollout_s | train_s | episodes | mean_ret | "
        "loss | policy | value | entropy | ev | approx_kl | clip_frac"
    )

    t_start = time.perf_counter()
    total_env_steps = 0
    finished_returns_window: list[float] = []

    for update_idx in range(1, cfg.total_updates + 1):
        # ------- rollout -------
        t_rollout = time.perf_counter()
        rollout_records = []
        for _ in range(cfg.rollout_steps):
            states = maybe_spawn_comets_host(states, cfg)
            rng, sub = jax.random.split(rng)
            states, rec, rng = rollout_step(states, params, sub, learner_players)
            rollout_records.append(rec)
            done_np = np.asarray(rec["done"])
            reward_np = np.asarray(rec["reward"])
            for i in range(cfg.num_envs):
                if done_np[i]:
                    finished_returns_window.append(float(reward_np[i]))
            if done_np.any():
                states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
                # Update learner_players only for envs that just reset.
                lp_np = np.asarray(learner_players)
                lp_np = np.where(done_np, new_lp, lp_np)
                learner_players = jnp.asarray(lp_np)

        rollout_s = time.perf_counter() - t_rollout
        total_env_steps += cfg.rollout_steps * cfg.num_envs

        # ------- bootstrap value for GAE -------
        feats_boot = jax.vmap(encode_observation, in_axes=(0, 0))(states, learner_players)
        out_boot = model.apply(params, **feats_boot)
        next_value = out_boot.value                                   # (B,)

        # ------- assemble (B, T) tensors -------
        # rollout_records is a list of dicts; stack along T axis.
        def stack_t(key, leaves):
            return jnp.stack([r[key] for r in leaves], axis=1)        # (B, T, ...)

        rewards = stack_t("reward", rollout_records)                   # (B, T)
        dones = stack_t("done", rollout_records)                       # (B, T)
        values = stack_t("value", rollout_records)                     # (B, T)

        adv, ret = compute_gae(rewards, values, dones, next_value, cfg.gamma, cfg.gae_lambda)
        # Normalize advantages.
        adv_mean = jnp.mean(adv)
        adv_std = jnp.std(adv) + 1e-8
        adv = (adv - adv_mean) / adv_std

        # Flatten to (N = B*T, ...).
        def flatten(arr):
            shape = arr.shape
            return arr.reshape((shape[0] * shape[1],) + shape[2:])

        flat = {}
        for k in (
            "planet_features", "planet_mask",
            "fleet_features", "fleet_mask",
            "global_features",
            "target_idx", "bucket_idx", "log_prob", "source_valid",
            "target_has_bucket", "bucket_valid",
        ):
            flat[k] = flatten(stack_t(k, rollout_records))
        flat["old_log_prob"] = flat.pop("log_prob")
        flat["advantages"] = flatten(adv)
        flat["returns"] = flatten(ret)

        n_rows = flat["advantages"].shape[0]

        # ------- PPO update -------
        t_train = time.perf_counter()
        metrics_accum = {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
        }
        opt_steps = 0
        for _ in range(cfg.epochs):
            perm = np.random.permutation(n_rows)
            for start in range(0, n_rows, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb = {k: v[idx] for k, v in flat.items()}
                params, opt_state, loss_val, m = update_step(params, opt_state, mb)
                metrics_accum["loss"] += float(loss_val)
                for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
                    metrics_accum[k] += float(m[k])
                opt_steps += 1

        train_s = time.perf_counter() - t_train
        if opt_steps:
            for k in metrics_accum:
                metrics_accum[k] /= opt_steps

        ev = float(explained_variance(flat["returns"], flat["advantages"] + flat["returns"] - flat["advantages"]))  # ≈ EV(returns, returns) sanity
        # Better: compute new values on (subset of) batch — quick approximation:
        idx = np.random.choice(n_rows, size=min(1024, n_rows), replace=False)
        sub = {k: v[idx] for k, v in flat.items()}
        v_sub = model.apply(
            params,
            planet_features=sub["planet_features"], planet_mask=sub["planet_mask"],
            fleet_features=sub["fleet_features"], fleet_mask=sub["fleet_mask"],
            global_features=sub["global_features"],
        ).value
        ev = float(explained_variance(sub["returns"], v_sub))

        elapsed = time.perf_counter() - t_start
        upd_s = update_idx / elapsed
        env_sps = total_env_steps / elapsed

        mean_ret = float(np.mean(finished_returns_window[-50:])) if finished_returns_window else float("nan")
        episodes = len(finished_returns_window)

        if update_idx % cfg.log_every == 0:
            print(
                f"{update_idx:6d} | {upd_s:5.2f} | {env_sps:7.0f} | {rollout_s:8.1f} | "
                f"{train_s:6.1f} | {episodes:7d} | "
                f"{mean_ret:+.3f} | "
                f"{metrics_accum['loss']:.4f} | {metrics_accum['policy_loss']:+.4f} | "
                f"{metrics_accum['value_loss']:.4f} | {metrics_accum['entropy']:.3f} | "
                f"{ev:+.3f} | {metrics_accum['approx_kl']:.5f} | {metrics_accum['clip_fraction']:.3f}"
            )

        if update_idx % cfg.checkpoint_every == 0 or update_idx == cfg.total_updates:
            blob = np.frombuffer(flax.serialization.to_bytes(params), dtype=np.uint8)
            np.savez(save_dir / "ckpt_last.npz", update=update_idx, params=blob)
            np.savez(save_dir / f"ckpt_{update_idx:06d}.npz", update=update_idx, params=blob)

    total_elapsed = time.perf_counter() - t_start
    print(f"Done. total_env_steps={total_env_steps} elapsed={total_elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_transformer.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
