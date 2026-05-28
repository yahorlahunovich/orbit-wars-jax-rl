"""PPO trainer for the JAX Transformer Orbit Wars policy.

End-to-end design:

- `num_envs` envs are stacked via `tree_map(jnp.stack)` and stepped with
  `jax.vmap(step_jit)` inside a `jax.lax.scan` of length `rollout_steps`.
- Self-play: both players use the same `params`. Per env we randomly assign
  which player is the learner at reset time; PPO trains on the learner's
  decision rows only.
- Optional curriculum: train vs `versions/kaggle700_current_heuristic` until a
  rolling win-rate threshold is met, then continue self-play.
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
from orbit_wars.heuristic_opponent import batched_heuristic_actions, load_heuristic_agent
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

    # Opponent / curriculum
    opponent: str = "selfplay"  # selfplay | heuristic | curriculum
    heuristic_win_rate: float = 0.35
    heuristic_window_episodes: int = 80
    heuristic_path: str | None = None


def load_config(path: str | Path) -> TrainConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    env = data.get("env", {})
    model = data.get("model", {})
    ppo = data.get("ppo", {})
    training = data.get("training", {})
    heur_path = training.get("heuristic_path")
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
        opponent=str(training.get("opponent", "selfplay")),
        heuristic_win_rate=float(training.get("heuristic_win_rate", 0.35)),
        heuristic_window_episodes=int(training.get("heuristic_window_episodes", 80)),
        heuristic_path=None if heur_path in (None, "null") else str(heur_path),
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


def _gather_by_player(zero_t, one_t, learner_players: jnp.ndarray):
    """Select per-env rows from player-0 or player-1 tensors."""
    lp = learner_players.astype(jnp.bool_)
    if zero_t.ndim == 1:
        return jnp.where(lp, one_t, zero_t)
    if zero_t.ndim == 2:
        return jnp.where(lp[:, None], one_t, zero_t)
    if zero_t.ndim == 3:
        return jnp.where(lp[:, None, None], one_t, zero_t)
    return jnp.where(lp[:, None, None, None], one_t, zero_t)


def sample_both_players_factory(model: PlanetPolicy):
    """Sample policy actions for both seats (same params, independent RNG)."""

    @jax.jit
    def sample(states: OrbitWarsState, params, rng):
        rng, k0, k1 = jax.random.split(rng, 3)
        feats0 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(0))
        out0 = model.apply(params, **feats0)
        grid0 = jax.vmap(compose_action_grid, in_axes=(0, None))(states, jnp.int32(0))
        s0 = sample_actions(k0, out0.target_logits, out0.bucket_logits, grid0)
        a0, m0 = pack_padded_actions(s0["target_idx"], s0["bucket_idx"], s0["source_valid"], grid0)

        feats1 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(1))
        out1 = model.apply(params, **feats1)
        grid1 = jax.vmap(compose_action_grid, in_axes=(0, None))(states, jnp.int32(1))
        s1 = sample_actions(k1, out1.target_logits, out1.bucket_logits, grid1)
        a1, m1 = pack_padded_actions(s1["target_idx"], s1["bucket_idx"], s1["source_valid"], grid1)
        return (a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, rng)

    return sample


def learner_record_from_samples(
    learner_players: jnp.ndarray,
    s0,
    s1,
    out0,
    out1,
    grid0,
    grid1,
    feats0,
    feats1,
    new_states: OrbitWarsState,
) -> dict:
    learner_feats = jax.tree_util.tree_map(
        lambda z, o: _gather_by_player(z, o, learner_players), feats0, feats1,
    )
    learner_value = _gather_by_player(out0.value, out1.value, learner_players)
    target_idx = _gather_by_player(s0["target_idx"], s1["target_idx"], learner_players)
    bucket_idx = _gather_by_player(s0["bucket_idx"], s1["bucket_idx"], learner_players)
    log_prob = _gather_by_player(s0["log_prob"], s1["log_prob"], learner_players)
    source_valid = _gather_by_player(s0["source_valid"], s1["source_valid"], learner_players)
    target_has_bucket = _gather_by_player(
        jnp.any(grid0["full_valid"], axis=-1),
        jnp.any(grid1["full_valid"], axis=-1),
        learner_players,
    )
    bucket_valid = _gather_by_player(grid0["full_valid"], grid1["full_valid"], learner_players)

    reward = jnp.where(
        new_states.done & (learner_players == 0),
        new_states.rewards[:, 0],
        jnp.where(
            new_states.done & (learner_players == 1),
            new_states.rewards[:, 1],
            jnp.zeros_like(new_states.rewards[:, 0]),
        ),
    )
    opp_reward = jnp.where(
        new_states.done & (learner_players == 0),
        new_states.rewards[:, 1],
        jnp.where(
            new_states.done & (learner_players == 1),
            new_states.rewards[:, 0],
            jnp.zeros_like(new_states.rewards[:, 1]),
        ),
    )
    return {
        "planet_features": learner_feats["planet_features"],
        "planet_mask": learner_feats["planet_mask"],

        "target_idx": target_idx,
        "bucket_idx": bucket_idx,
        "log_prob": log_prob,
        "source_valid": source_valid,
        "target_has_bucket": target_has_bucket,
        "bucket_valid": bucket_valid,
        "value": learner_value,
        "reward": reward,
        "opp_reward": opp_reward,
        "done": new_states.done,
    }


def rollout_step_selfplay_factory(model: PlanetPolicy):
    sample = sample_both_players_factory(model)
    step_jit = __import__("orbit_wars.step", fromlist=["step_jit"]).step_jit

    @jax.jit
    def step_one(states: OrbitWarsState, params, rng, learner_players):
        a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, rng = sample(
            states, params, rng,
        )
        new_states = jax.vmap(step_jit)(states, a0, a1, m0, m1)
        record = learner_record_from_samples(
            learner_players, s0, s1, out0, out1, grid0, grid1, feats0, feats1, new_states,
        )
        return new_states, record, rng

    return step_one


def rollout_step_vs_heuristic_factory(model: PlanetPolicy):
    """Policy learner + frozen heuristic opponent (host-side opponent actions)."""
    sample = sample_both_players_factory(model)
    step_jit = __import__("orbit_wars.step", fromlist=["step_jit"]).step_jit

    def step_one(states: OrbitWarsState, params, rng, learner_players, heuristic_agent):
        a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, rng = sample(
            states, params, rng,
        )
        lp_np = np.asarray(learner_players, dtype=np.int32)
        opp_np = 1 - lp_np
        ha0, hm0, ha1, hm1 = batched_heuristic_actions(states, opp_np, heuristic_agent)

        is_learner_p0 = (learner_players == 0)
        final_a0 = jnp.where(is_learner_p0[:, None, None], a0, ha0)
        final_a1 = jnp.where(is_learner_p0[:, None, None], ha1, a1)
        final_m0 = jnp.where(is_learner_p0[:, None], m0, hm0)
        final_m1 = jnp.where(is_learner_p0[:, None], hm1, m1)

        new_states = jax.vmap(step_jit)(states, final_a0, final_a1, final_m0, final_m1)
        record = learner_record_from_samples(
            learner_players, s0, s1, out0, out1, grid0, grid1, feats0, feats1, new_states,
        )
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

    rollout_selfplay = rollout_step_selfplay_factory(model)
    rollout_vs_heuristic = rollout_step_vs_heuristic_factory(model)
    update_step = make_update_step(model, optimizer, cfg)

    opponent_mode = cfg.opponent.lower()
    if opponent_mode not in ("selfplay", "heuristic", "curriculum"):
        raise ValueError(f"unknown opponent mode: {cfg.opponent}")
    active_mode = "heuristic" if opponent_mode in ("heuristic", "curriculum") else "selfplay"
    curriculum_switched = False

    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = save_dir / "training.log"
    log_file = log_file_path.open("a", encoding="utf-8")

    def log_print(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    heuristic_agent = None
    if active_mode == "heuristic":
        heur_path = Path(cfg.heuristic_path) if cfg.heuristic_path else None
        heuristic_agent = load_heuristic_agent(heur_path)
        log_print(f"Heuristic opponent loaded from {heur_path or 'default'}")

    seed_base = cfg.seed * 10000 + 1
    states, learner_players_np = make_initial_states(cfg, seed_base)
    learner_players = jnp.asarray(learner_players_np)
    next_seed = seed_base + cfg.num_envs

    log_print(
        f"JAX devices: {jax.devices()} | envs={cfg.num_envs} rollout={cfg.rollout_steps} "
        f"updates={cfg.total_updates} opponent={opponent_mode} active={active_mode}"
    )
    log_print(
        "update |     mode | lrnr_wr | W-L-D | episodes | mean_ret | env_sps | "
        "  loss | pol_loss | val_loss | entropy |     ev | approx_kl | clip_fr"
    )

    t_start = time.perf_counter()
    total_env_steps = 0
    finished_returns_window: list[float] = []
    heuristic_returns_window: list[float] = []
    learner_wins = learner_losses = learner_draws = 0

    for update_idx in range(1, cfg.total_updates + 1):
        t_rollout = time.perf_counter()
        rollout_records = []
        for _ in range(cfg.rollout_steps):
            states = maybe_spawn_comets_host(states, cfg)
            rng, sub = jax.random.split(rng)
            if active_mode == "selfplay":
                states, rec, rng = rollout_selfplay(states, params, sub, learner_players)
            else:
                states, rec, rng = rollout_vs_heuristic(
                    states, params, sub, learner_players, heuristic_agent,
                )
            rollout_records.append(rec)
            done_np = np.asarray(rec["done"])
            reward_np = np.asarray(rec["reward"])
            opp_reward_np = np.asarray(rec.get("opp_reward", np.zeros_like(reward_np)))
            for i in range(cfg.num_envs):
                if done_np[i]:
                    r = float(reward_np[i])
                    opp_r = float(opp_reward_np[i])
                    finished_returns_window.append(r)
                    if active_mode == "heuristic":
                        heuristic_returns_window.append(r)
                        if r > opp_r:
                            learner_wins += 1
                        elif r < opp_r:
                            learner_losses += 1
                        else:
                            learner_draws += 1
            if done_np.any():
                states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
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

        ).value
        ev = float(explained_variance(sub["returns"], v_sub))

        elapsed = time.perf_counter() - t_start
        env_sps = total_env_steps / elapsed

        mean_ret = float(np.mean(finished_returns_window[-50:])) if finished_returns_window else float("nan")
        episodes = len(finished_returns_window)

        window = heuristic_returns_window[-cfg.heuristic_window_episodes :]
        learner_wr = float(np.mean([1.0 if r > 0 else 0.0 for r in window])) if window else float("nan")
        wld = f"{learner_wins}-{learner_losses}-{learner_draws}"

        if update_idx % cfg.log_every == 0:
            log_print(
                f"{update_idx:6d} | {active_mode:8s} | "
                f"{learner_wr if active_mode == 'heuristic' else float('nan'):7.1%} | "
                f"{wld if active_mode == 'heuristic' else 'n/a':>5s} | "
                f"{episodes:7d} | {mean_ret:+.3f} | {env_sps:7.0f} | "
                f"{metrics_accum['loss']:.4f} | {metrics_accum['policy_loss']:+.4f} | "
                f"{metrics_accum['value_loss']:.4f} | {metrics_accum['entropy']:.3f} | "
                f"{ev:+.3f} | {metrics_accum['approx_kl']:.5f} | {metrics_accum['clip_fraction']:.3f}"
            )
            learner_wins = learner_losses = learner_draws = 0

        if (
            opponent_mode == "curriculum"
            and active_mode == "heuristic"
            and not curriculum_switched
            and len(window) >= cfg.heuristic_window_episodes
            and learner_wr >= cfg.heuristic_win_rate
        ):
            active_mode = "selfplay"
            curriculum_switched = True
            log_print("=" * 72)
            log_print(
                f"CURRICULUM SWITCH at update {update_idx}: "
                f"heuristic win rate {learner_wr:.1%} >= {cfg.heuristic_win_rate:.1%}. "
                f"Continuing with self-play for remaining updates."
            )
            log_print("=" * 72)

        if update_idx % cfg.checkpoint_every == 0 or update_idx == cfg.total_updates:
            blob = np.frombuffer(flax.serialization.to_bytes(params), dtype=np.uint8)
            np.savez(save_dir / "ckpt_last.npz", update=update_idx, params=blob)
            np.savez(save_dir / f"ckpt_{update_idx:06d}.npz", update=update_idx, params=blob)

    total_elapsed = time.perf_counter() - t_start
    log_print(f"Done. total_env_steps={total_env_steps} elapsed={total_elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_transformer.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
