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
import functools
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
from orbit_wars.decode import INTERCEPT_ITERATIONS
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
    intercept_iterations: int = INTERCEPT_ITERATIONS
    enable_planet_block: bool = True
    enable_incoming_projection: bool = True

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
    weight_decay: float = 0.0

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
        intercept_iterations=int(env.get("intercept_iterations", INTERCEPT_ITERATIONS)),
        enable_planet_block=bool(env.get("enable_planet_block", True)),
        enable_incoming_projection=bool(env.get("enable_incoming_projection", True)),
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
        weight_decay=float(ppo.get("weight_decay", 0.0)),
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


def sample_both_players_factory(model: PlanetPolicy, grid_fn):
    """Sample policy actions for both seats. Uses params for learner, opp_params for opponent."""

    @jax.jit
    def sample(states: OrbitWarsState, params, opp_params, rng, learner_players):
        rng, k0, k1 = jax.random.split(rng, 3)
        
        feats0 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(0))
        feats1 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(1))
        
        def _gather_feats(f0, f1, is_p0):
            # Expands the (B,) mask to the shape of f0/f1 for jnp.where
            mask = is_p0
            for _ in range(f0.ndim - 1):
                mask = mask[..., None]
            return jnp.where(mask, f0, f1)

        is_learner_p0 = (learner_players == 0)
        is_opp_p0 = (learner_players == 1)

        feats_learner = jax.tree_util.tree_map(lambda f0, f1: _gather_feats(f0, f1, is_learner_p0), feats0, feats1)
        feats_opp = jax.tree_util.tree_map(lambda f0, f1: _gather_feats(f0, f1, is_opp_p0), feats0, feats1)

        out_learner = model.apply(params, **feats_learner)
        out_opp = model.apply(opp_params, **feats_opp)

        out0 = jax.tree_util.tree_map(lambda l, o: _gather_feats(l, o, is_learner_p0), out_learner, out_opp)
        out1 = jax.tree_util.tree_map(lambda l, o: _gather_feats(l, o, is_opp_p0), out_learner, out_opp)

        grid0 = jax.vmap(grid_fn, in_axes=(0, None))(states, jnp.int32(0))
        s0 = sample_actions(k0, out0.target_logits, out0.bucket_logits, grid0)
        a0, m0 = pack_padded_actions(s0["target_idx"], s0["bucket_idx"], s0["source_valid"], grid0)

        grid1 = jax.vmap(grid_fn, in_axes=(0, None))(states, jnp.int32(1))
        s1 = sample_actions(k1, out1.target_logits, out1.bucket_logits, grid1)
        a1, m1 = pack_padded_actions(s1["target_idx"], s1["bucket_idx"], s1["source_valid"], grid1)
        
        return (a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, rng)

    return sample


def sample_learner_factory(model: PlanetPolicy, grid_fn):
    """Sample policy actions for the learner seat only (player varies per env)."""

    @jax.jit
    def sample(states: OrbitWarsState, params, rng, learner_players):
        rng, k0 = jax.random.split(rng)
        feats = jax.vmap(encode_observation, in_axes=(0, 0))(states, learner_players)
        out = model.apply(params, **feats)
        grid = jax.vmap(grid_fn, in_axes=(0, 0))(states, learner_players)
        sampled = sample_actions(k0, out.target_logits, out.bucket_logits, grid)
        actions, mask = pack_padded_actions(
            sampled["target_idx"], sampled["bucket_idx"], sampled["source_valid"], grid,
        )
        return actions, mask, sampled, out, grid, feats, rng

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


def learner_record_from_single(
    learner_players: jnp.ndarray,
    sampled,
    out,
    grid,
    feats,
    new_states: OrbitWarsState,
) -> dict:
    target_has_bucket = jnp.any(grid["full_valid"], axis=-1)
    bucket_valid = grid["full_valid"]

    batch = jnp.arange(new_states.rewards.shape[0])
    lp = learner_players.astype(jnp.int32)
    opp = (1 - lp).astype(jnp.int32)
    reward = new_states.rewards[batch, lp]
    opp_reward = new_states.rewards[batch, opp]

    reward = jnp.where(new_states.done, reward, jnp.zeros_like(reward))
    opp_reward = jnp.where(new_states.done, opp_reward, jnp.zeros_like(opp_reward))

    return {
        "planet_features": feats["planet_features"],
        "planet_mask": feats["planet_mask"],
        "target_idx": sampled["target_idx"],
        "bucket_idx": sampled["bucket_idx"],
        "log_prob": sampled["log_prob"],
        "source_valid": sampled["source_valid"],
        "target_has_bucket": target_has_bucket,
        "bucket_valid": bucket_valid,
        "value": out.value,
        "reward": reward,
        "opp_reward": opp_reward,
        "done": new_states.done,
    }


def rollout_step_selfplay_factory(model: PlanetPolicy, grid_fn):
    sample = sample_both_players_factory(model, grid_fn)
    step_jit = __import__("orbit_wars.step", fromlist=["step_jit"]).step_jit

    @jax.jit
    def step_one(states: OrbitWarsState, params, opp_params, rng, learner_players, reset_pool):
        rng, k_sample, k_pool, k_lp = jax.random.split(rng, 4)
        a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, k_sample = sample(
            states, params, opp_params, k_sample, learner_players
        )
        new_states = jax.vmap(step_jit)(states, a0, a1, m0, m1)
        record = learner_record_from_samples(
            learner_players, s0, s1, out0, out1, grid0, grid1, feats0, feats1, new_states,
        )
        
        dones = new_states.done
        
        # Auto-reset
        pool_size = reset_pool.step.shape[0]
        pool_indices = jax.random.randint(k_pool, (states.step.shape[0],), 0, pool_size)
        fresh_states = jax.tree_util.tree_map(lambda p: p[pool_indices], reset_pool)
        
        next_states = jax.tree_util.tree_map(
            lambda new_s, fresh_s: jnp.where(
                dones[(...,) + (None,) * (new_s.ndim - 1)], fresh_s, new_s
            ),
            new_states, fresh_states
        )
        
        new_learner_players = jax.random.randint(k_lp, (states.step.shape[0],), 0, 2)
        next_learner_players = jnp.where(dones, new_learner_players, learner_players)
        
        return next_states, record, rng, next_learner_players

    return step_one


def rollout_step_vs_heuristic_factory(model: PlanetPolicy, grid_fn):
    """Policy learner + frozen heuristic opponent (host-side opponent actions)."""
    sample = sample_learner_factory(model, grid_fn)
    step_jit = __import__("orbit_wars.step", fromlist=["step_jit"]).step_jit

    def step_one(
        states: OrbitWarsState,
        params,
        rng,
        learner_players,
        opponent_players_np: np.ndarray,
        heuristic_agent,
    ):
        actions, mask, sampled, out, grid, feats, rng = sample(states, params, rng, learner_players)
        ha0, hm0, ha1, hm1 = batched_heuristic_actions(states, opponent_players_np, heuristic_agent)

        is_learner_p0 = (learner_players == 0)
        final_a0 = jnp.where(is_learner_p0[:, None, None], actions, ha0)
        final_a1 = jnp.where(is_learner_p0[:, None, None], ha1, actions)
        final_m0 = jnp.where(is_learner_p0[:, None], mask, hm0)
        final_m1 = jnp.where(is_learner_p0[:, None], hm1, mask)

        new_states = jax.vmap(step_jit)(states, final_a0, final_a1, final_m0, final_m1)
        record = learner_record_from_single(learner_players, sampled, out, grid, feats, new_states)
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
    # Total optimizer steps = total_updates * epochs * ceil(n_rows / minibatch_size)
    n_rows = cfg.num_envs * cfg.rollout_steps
    steps_per_epoch = (n_rows + cfg.minibatch_size - 1) // cfg.minibatch_size
    total_steps = cfg.total_updates * cfg.epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=cfg.lr_start,
        decay_steps=total_steps,
        alpha=cfg.lr_end / max(cfg.lr_start, 1e-12),
    )
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adamw(schedule, weight_decay=cfg.weight_decay)), schedule


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
    opp_params = params
    optimizer, _ = make_optimizer(cfg)
    opt_state = optimizer.init(params)

    grid_fn = functools.partial(
        compose_action_grid,
        intercept_iterations=cfg.intercept_iterations,
        enable_planet_block=cfg.enable_planet_block,
        enable_incoming_projection=cfg.enable_incoming_projection,
    )
    rollout_selfplay = rollout_step_selfplay_factory(model, grid_fn)
    rollout_vs_heuristic = rollout_step_vs_heuristic_factory(model, grid_fn)
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

    # Pre-generate reset pool for auto-reset
    pool_size = max(256, cfg.num_envs * 4)
    log_print(f"Generating reset pool of size {pool_size}...")
    reset_pool_states, _ = make_initial_states(cfg, cfg.seed + 100000)
    # We need a pool of `pool_size`. make_initial_states uses `num_envs` so we just call it with a custom config.
    import dataclasses
    cfg_pool = dataclasses.replace(cfg, num_envs=pool_size)
    reset_pool, _ = make_initial_states(cfg_pool, cfg.seed + 100000)
    
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
            if active_mode != "selfplay":
                states = maybe_spawn_comets_host(states, cfg)
            rng, sub = jax.random.split(rng)
            if active_mode == "selfplay":
                states, rec, rng, learner_players = rollout_selfplay(states, params, opp_params, sub, learner_players, reset_pool)
            else:
                opp_np = 1 - learner_players_np
                states, rec, rng = rollout_vs_heuristic(
                    states, params, sub, learner_players, opp_np, heuristic_agent,
                )
            rollout_records.append(rec)
            
            if active_mode != "selfplay":
                done_np = np.asarray(rec["done"])
                if done_np.any():
                    reward_np = np.asarray(rec["reward"])
                    finished_returns_window.extend(reward_np[done_np].tolist())
                    opp_reward_np = np.asarray(rec.get("opp_reward", np.zeros_like(reward_np)))
                    heuristic_returns_window.extend(reward_np[done_np].tolist())
                    wins = np.sum((reward_np > opp_reward_np) & done_np)
                    losses = np.sum((reward_np < opp_reward_np) & done_np)
                    draws = np.sum((reward_np == opp_reward_np) & done_np)
                    learner_wins += int(wins)
                    learner_losses += int(losses)
                    learner_draws += int(draws)
    
                    states, next_seed, new_lp = reset_done_envs(states, done_np, next_seed, cfg)
                    learner_players_np = np.where(done_np, new_lp, learner_players_np)
                    learner_players = jnp.asarray(learner_players_np)

        # For selfplay, process dones after the rollout loop to avoid blocking GPU
        if active_mode == "selfplay":
            # Just extract the data once it's all done
            dones_batch = jnp.stack([r["done"] for r in rollout_records], axis=1)
            rewards_batch = jnp.stack([r["reward"] for r in rollout_records], axis=1)
            opp_rewards_batch = jnp.stack([r["opp_reward"] for r in rollout_records], axis=1)
            done_mask = np.asarray(dones_batch)
            reward_vals = np.asarray(rewards_batch)
            opp_reward_vals = np.asarray(opp_rewards_batch)
            if done_mask.any():
                finished_returns_window.extend(reward_vals[done_mask].tolist())
                heuristic_returns_window.extend(reward_vals[done_mask].tolist())
                wins = np.sum((reward_vals > opp_reward_vals) & done_mask)
                losses = np.sum((reward_vals < opp_reward_vals) & done_mask)
                draws = np.sum((reward_vals == opp_reward_vals) & done_mask)
                learner_wins += int(wins)
                learner_losses += int(losses)
                learner_draws += int(draws)

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

        # Calculate winrates for the dual thresholds (100 and 200 games)
        # Note: heuristic_returns_window is cleared every time the opponent updates.
        win_100 = heuristic_returns_window[-100:]
        wr_100 = float(np.mean([1.0 if r > 0 else 0.0 for r in win_100])) if len(win_100) >= 100 else float("nan")
        
        win_200 = heuristic_returns_window[-200:]
        wr_200 = float(np.mean([1.0 if r > 0 else 0.0 for r in win_200])) if len(win_200) >= 200 else float("nan")
        
        # Display winrate against current opponent in logs (using largest available window up to 100)
        display_window = heuristic_returns_window[-100:]
        learner_wr = float(np.mean([1.0 if r > 0 else 0.0 for r in display_window])) if display_window else float("nan")
        
        wld = f"{learner_wins}-{learner_losses}-{learner_draws}"

        if update_idx % cfg.log_every == 0:
            log_print(
                f"{update_idx:6d} | {active_mode:8s} | "
                f"{learner_wr:7.1%} | "
                f"{wld:>5s} | "
                f"{episodes:7d} | {mean_ret:+.3f} | {env_sps:7.0f} | "
                f"{metrics_accum['loss']:.4f} | {metrics_accum['policy_loss']:+.4f} | "
                f"{metrics_accum['value_loss']:.4f} | {metrics_accum['entropy']:.3f} | "
                f"{ev:+.3f} | {metrics_accum['approx_kl']:.5f} | {metrics_accum['clip_fraction']:.3f}"
            )
            learner_wins = learner_losses = learner_draws = 0

        update_opp = False
        if active_mode == "selfplay" and update_idx % 5 == 0:
            if not np.isnan(wr_100) and wr_100 > 0.56:
                log_print(f"Update {update_idx}: Self-play winrate {wr_100:.1%} > 56% (100 games). Updating opponent parameters.")
                update_opp = True
            elif not np.isnan(wr_200) and wr_200 > 0.54:
                log_print(f"Update {update_idx}: Self-play winrate {wr_200:.1%} > 54% (200 games). Updating opponent parameters.")
                update_opp = True
        
        if update_opp:
            opp_params = params
            heuristic_returns_window.clear()

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
