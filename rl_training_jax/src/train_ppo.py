"""JAX PPO training for Orbit Wars."""

from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from orbit_wars import (
    BUCKET_COUNT,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    PLANET_FEATURE_DIM,
    OrbitWarsState,
    compose_action_grid,
    encode_observation,
    reset,
)
from orbit_wars.rollout import pack_padded_actions, sample_actions
from ppo import compute_gae, explained_variance
from policy import PlanetPolicy


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    run_name: str = "jax_ppo_transformer"
    save_dir: str = "artifacts"

    # Env
    num_envs: int = 32
    episode_steps: int = 500
    rollout_steps: int = 32

    # Model
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    bucket_count: int = 8
    weight_decay: float = 1e-4

    # PPO
    total_updates: int = 5000
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.01
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_coef: float = 0.2
    ent_coef: float = 0.05  # Increased to fight policy collapse
    vf_coef: float = 0.5
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    lr_end: float = 1e-6
    lr_warmup_updates: int = 100
    lr_total_updates: int = 1000
    max_grad_norm: float = 0.5

    # Opponent
    opponent: str = "selfplay"  # selfplay | heuristic | curriculum
    heuristic_win_rate: float = 0.6
    heuristic_window_episodes: int = 100

    log_every: int = 5
    checkpoint_every: int = 100


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    env = data.get("env", {})
    model = data.get("model", {})
    ppo = data.get("ppo", {})
    opp = data.get("opponent", {})

    return TrainConfig(
        seed=int(data.get("seed", 0)),
        run_name=str(data.get("run_name", "jax_ppo_transformer")),
        save_dir=str(data.get("save_dir", "artifacts")),
        num_envs=int(env.get("num_envs", 32)),
        episode_steps=int(env.get("episode_steps", 500)),
        rollout_steps=int(env.get("rollout_steps", 32)),
        d_model=int(model.get("d_model", 96)),
        num_heads=int(model.get("num_heads", 4)),
        num_layers=int(model.get("num_layers", 3)),
        bucket_count=int(model.get("bucket_count", 8)),
        weight_decay=float(model.get("weight_decay", 1e-4)),
        total_updates=int(ppo.get("total_updates", 5000)),
        train_pi_iters=int(ppo.get("train_pi_iters", 80)),
        train_v_iters=int(ppo.get("train_v_iters", 80)),
        target_kl=float(ppo.get("target_kl", 0.01)),
        minibatch_size=int(ppo.get("minibatch_size", 256)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.97)),
        clip_coef=float(ppo.get("clip_coef", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.05)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        pi_lr=float(ppo.get("pi_lr", 3e-4)),
        vf_lr=float(ppo.get("vf_lr", 1e-3)),
        lr_end=float(ppo.get("lr_end", 1e-6)),
        lr_warmup_updates=int(ppo.get("lr_warmup_updates", 100)),
        lr_total_updates=int(ppo.get("lr_total_updates", 1000)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        opponent=str(data.get("opponent", "selfplay")),
        heuristic_win_rate=float(opp.get("win_rate", 0.6)),
        heuristic_window_episodes=int(opp.get("window_episodes", 100)),
        log_every=int(data.get("log_every", 5)),
        checkpoint_every=int(data.get("checkpoint_every", 100)),
    )


# ---------------------------------------------------------------------------
# Batched heuristic helpers
# ---------------------------------------------------------------------------


def _load_heuristic_agent():
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "versions/kaggle700_current_heuristic/main.py"
    heur_root = path.parent
    sys.path.insert(0, str(heur_root))
    spec = importlib.util.spec_from_file_location("heuristic_main", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.agent


def batched_heuristic_actions(states: OrbitWarsState, players: np.ndarray, agent):
    """Call the heuristic agent for every env in the batch via Python loop."""
    from orbit_wars.convert import state_to_observation_dict
    from orbit_wars.step import _list_action_to_padded

    b = states.step.shape[0]
    a0_list, a1_list = [], []
    m0_list, m1_list = [], []

    for i in range(b):
        single_state = jax.tree_util.tree_map(lambda x: x[i], states)
        # Agent expects its own seat in obs["player"].
        p = int(players[i])
        obs = state_to_observation_dict(single_state, player=p)
        moves = agent(obs)

        # Convert list moves to padded (MAX_MOVES, 3) + mask.
        row, mask = _list_action_to_padded(moves)
        if p == 0:
            a0_list.append(row)
            m0_list.append(mask)
            a1_list.append(jnp.zeros_like(row))
            m1_list.append(jnp.zeros_like(mask))
        else:
            a0_list.append(jnp.zeros_like(row))
            m0_list.append(jnp.zeros_like(mask))
            a1_list.append(row)
            m1_list.append(mask)

    return (
        jnp.stack(a0_list),
        jnp.stack(m0_list),
        jnp.stack(a1_list),
        jnp.stack(m1_list),
    )


# ---------------------------------------------------------------------------
# Rollout / Sampling logic
# ---------------------------------------------------------------------------


def sample_both_players_factory(model: PlanetPolicy, grid_fn):
    """Sample policy actions for both seats. Uses params for learner, opp_params for opponent."""

    @jax.jit
    def sample(states: OrbitWarsState, params, opp_params, rng, learner_players):
        rng, k0, k1 = jax.random.split(rng, 3)
        
        feats0 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(0))
        feats1 = jax.vmap(encode_observation, in_axes=(0, None))(states, jnp.int32(1))
        
        def _gather_feats(f0, f1, is_p0):
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
        a0, m0, em0 = pack_padded_actions(s0["target_idx"], s0["bucket_idx"], s0["source_valid"], grid0)

        grid1 = jax.vmap(grid_fn, in_axes=(0, None))(states, jnp.int32(1))
        s1 = sample_actions(k1, out1.target_logits, out1.bucket_logits, grid1)
        a1, m1, em1 = pack_padded_actions(s1["target_idx"], s1["bucket_idx"], s1["source_valid"], grid1)
        
        return (a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, em0, em1, rng)

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
        actions, mask, executed_mask = pack_padded_actions(
            sampled["target_idx"], sampled["bucket_idx"], sampled["source_valid"], grid
        )
        return actions, mask, executed_mask, sampled, out, grid, feats, rng

    return sample


def _gather_by_player(one_t, zero_t, lp):
    """gather(p0_tensor, p1_tensor, learner_players)."""
    # lp is (B,) int32. Expand it to match the rank of one_t.
    mask = lp
    for _ in range(one_t.ndim - 1):
        mask = mask[..., None]
    return jnp.where(mask, zero_t, one_t)


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
    em0,
    em1,
    new_states: OrbitWarsState,
) -> dict:
    learner_feats = jax.tree_util.tree_map(
        lambda z, o: _gather_by_player(z, o, learner_players), feats0, feats1,
    )
    learner_value = _gather_by_player(out0.value, out1.value, learner_players)
    target_idx = _gather_by_player(s0["target_idx"], s1["target_idx"], learner_players)
    bucket_idx = _gather_by_player(s0["bucket_idx"], s1["bucket_idx"], learner_players)
    log_prob = _gather_by_player(s0["log_prob"], s1["log_prob"], learner_players)
    executed_mask = _gather_by_player(em0, em1, learner_players)
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
        "executed_mask": executed_mask,
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
    executed_mask,
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
        "executed_mask": executed_mask,
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
        a0, m0, a1, m1, s0, s1, out0, out1, grid0, grid1, feats0, feats1, em0, em1, k_sample = sample(
            states, params, opp_params, k_sample, learner_players
        )
        new_states = jax.vmap(step_jit)(states, a0, a1, m0, m1)
        record = learner_record_from_samples(
            learner_players, s0, s1, out0, out1, grid0, grid1, feats0, feats1, em0, em1, new_states,
        )
        
        dones = new_states.done
        
        # Auto-reset
        idx = jax.random.randint(k_pool, (states.step.shape[0],), 0, reset_pool.step.shape[0])
        resets = jax.tree_util.tree_map(lambda x: x[idx], reset_pool)
        next_states = jax.tree_util.tree_map(lambda s, r: jnp.where(dones, r, s), new_states, resets)
        
        # Cycle learner seats
        new_learner_players = jax.random.randint(k_lp, (states.step.shape[0],), 0, 2)
        next_learner_players = jnp.where(dones, new_learner_players, learner_players)
        
        return next_states, record, rng, next_learner_players

    return step_one


def rollout_step_vs_heuristic_factory(model: PlanetPolicy, grid_fn):
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
        actions, mask, executed_mask, sampled, out, grid, feats, rng = sample(states, params, rng, learner_players)
        ha0, hm0, ha1, hm1 = batched_heuristic_actions(states, opponent_players_np, heuristic_agent)

        is_learner_p0 = (learner_players == 0)
        final_a0 = jnp.where(is_learner_p0[:, None, None], actions, ha0)
        final_a1 = jnp.where(is_learner_p0[:, None, None], ha1, actions)
        final_m0 = jnp.where(is_learner_p0[:, None], mask, hm0)
        final_m1 = jnp.where(is_learner_p0[:, None], hm1, mask)

        new_states = jax.vmap(step_jit)(states, final_a0, final_a1, final_m0, final_m1)
        record = learner_record_from_single(
            learner_players, sampled, out, grid, feats, executed_mask, new_states,
        )
        return new_states, record, rng

    return step_one


def reset_done_envs(states: OrbitWarsState, dones_np: np.ndarray, next_seed: int, cfg: TrainConfig) -> tuple[OrbitWarsState, int, np.ndarray]:
    """Host-side resets for envs whose episodes ended. Returns (new_states, next_seed, new_lp)."""
    new_states_list = []
    new_lp_list = []
    for i in range(cfg.num_envs):
        if dones_np[i]:
            new_states_list.append(reset(next_seed, episode_steps=cfg.episode_steps))
            new_lp_list.append(next_seed % 2)
            next_seed += 1
        else:
            new_states_list.append(jax.tree_util.tree_map(lambda x: x[i], states))
            new_lp_list.append(-1) # Placeholder
    
    new_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_states_list)
    return new_states, next_seed, np.array(new_lp_list)


def maybe_spawn_comets_host(states: OrbitWarsState, cfg: TrainConfig) -> OrbitWarsState:
    from orbit_wars.comet import spawn_comet_for_state
    
    steps = np.asarray(states.step)
    COMET_SPAWN_STEPS = (50, 100, 150)
    
    new_states_list = []
    changed = False
    for i in range(cfg.num_envs):
        single = jax.tree_util.tree_map(lambda x: x[i], states)
        if steps[i] in COMET_SPAWN_STEPS:
            new_states_list.append(spawn_comet_for_state(single))
            changed = True
        else:
            new_states_list.append(single)
    
    if changed:
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_states_list)
    return states


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def make_optimizer(cfg: TrainConfig, params):
    n_rows = cfg.num_envs * cfg.rollout_steps
    steps_per_epoch = (n_rows + cfg.minibatch_size - 1) // cfg.minibatch_size
    
    # We estimate total optimizer steps based on the MAX iterations.
    # Note: policy might early stop, but value usually doesn't.
    steps_per_update = (cfg.train_pi_iters + cfg.train_v_iters) * steps_per_epoch
    
    warmup_steps = cfg.lr_warmup_updates * steps_per_update
    total_steps = cfg.lr_total_updates * steps_per_update

    def make_schedule(init_lr):
        return optax.warmup_linear_decay_schedule(
            init_value=0.0,
            peak_value=init_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=cfg.lr_end,
        )

    pi_schedule = make_schedule(cfg.pi_lr)
    vf_schedule = make_schedule(cfg.vf_lr)

    def label_fn(path, _):
        names = [p.key if hasattr(p, 'key') else str(p) for p in path]
        if 'value_head' in names:
            return 'vf'
        return 'pi'

    labels = jax.tree_util.tree_map_with_path(label_fn, params)
    
    optimizer = optax.multi_transform(
        {
            'pi': optax.adamw(pi_schedule, weight_decay=cfg.weight_decay),
            'vf': optax.adamw(vf_schedule, weight_decay=cfg.weight_decay),
        },
        labels
    )
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optimizer)
    return optimizer, pi_schedule, vf_schedule


def make_policy_update_step(model: PlanetPolicy, optimizer, cfg: TrainConfig):
    from ppo import policy_loss_fn
    @jax.jit
    def update(params, opt_state, batch):
        def loss(p):
            return policy_loss_fn(p, model.apply, batch, cfg.clip_coef, cfg.ent_coef)
        (l, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params)
        
        # Zero out value_head grads during policy update
        def zero_vf_grads(path, g):
            names = [p.key if hasattr(p, 'key') else str(p) for p in path]
            return jnp.zeros_like(g) if 'value_head' in names else g
        grads = jax.tree_util.tree_map_with_path(zero_vf_grads, grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l, metrics
    return update


def make_value_update_step(model: PlanetPolicy, optimizer, cfg: TrainConfig):
    from ppo import value_loss_fn
    @jax.jit
    def update(params, opt_state, batch):
        def loss(p):
            return value_loss_fn(p, model.apply, batch)
        l, grads = jax.value_and_grad(loss)(params)
        
        # Zero out policy grads during value update
        def zero_pi_grads(path, g):
            names = [p.key if hasattr(p, 'key') else str(p) for p in path]
            return g if 'value_head' in names else jnp.zeros_like(g)
        grads = jax.tree_util.tree_map_with_path(zero_pi_grads, grads)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l
    return update


def init_policy_params(rng, model: PlanetPolicy):
    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
    }
    return model.init(rng, **example), example


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
    optimizer, pi_sched, vf_sched = make_optimizer(cfg, params)
    opt_state = optimizer.init(params)

    grid_fn = functools.partial(
        compose_action_grid,
        sun_path_margin=1.5,
        path_planet_margin=1.0,
        intercept_iterations=5,
    )

    rollout_selfplay = rollout_step_selfplay_factory(model, grid_fn)
    rollout_vs_heuristic = rollout_step_vs_heuristic_factory(model, grid_fn)
    policy_update = make_policy_update_step(model, optimizer, cfg)
    value_update = make_value_update_step(model, optimizer, cfg)

    opponent_mode = cfg.opponent.lower()
    if opponent_mode not in ("selfplay", "heuristic", "curriculum"):
        raise ValueError(f"Invalid opponent mode: {opponent_mode}")

    active_mode = "heuristic" if opponent_mode == "curriculum" else opponent_mode
    heuristic_agent = _load_heuristic_agent() if active_mode == "heuristic" or opponent_mode == "curriculum" else None
    
    # Pre-generate reset states for selfplay
    print(f"Generating reset pool of size 256...")
    reset_pool_list = [reset(s, episode_steps=cfg.episode_steps) for s in range(256)]
    reset_pool = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *reset_pool_list)

    # Init envs
    states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[reset(s, episode_steps=cfg.episode_steps) for s in range(cfg.num_envs)])
    learner_players_np = np.array([i % 2 for i in range(cfg.num_envs)])
    learner_players = jnp.asarray(learner_players_np)
    next_seed = cfg.num_envs

    # Metrics
    finished_returns_window = []
    heuristic_returns_window = []
    curriculum_switched = False
    total_env_steps = 0
    learner_wins = learner_losses = learner_draws = 0

    log_print = print
    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "  update |   mode   | lrnr_wr | W-L-D | episodes | mean_ret | env_sps | "
        "pol_loss | val_loss | entropy |   ev   | approx_kl | clip_fr | pi_lr | vf_lr"
    )
    log_print(header)
    log_print("-" * len(header))

    t_start = time.perf_counter()
    
    n_rows_full = cfg.num_envs * cfg.rollout_steps
    steps_per_epoch = (n_rows_full + cfg.minibatch_size - 1) // cfg.minibatch_size
    steps_per_update = (cfg.train_pi_iters + cfg.train_v_iters) * steps_per_epoch

    for update_idx in range(1, cfg.total_updates + 1):
        # ------- Rollout -------
        rollout_records = []
        t_rollout = time.perf_counter()
        
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

        # Flatten to (N = B*T, ...).
        def flatten(arr):
            shape = arr.shape
            return arr.reshape((shape[0] * shape[1],) + shape[2:])

        flat = {}
        for k in (
            "planet_features", "planet_mask",
            "target_idx", "bucket_idx", "log_prob", "executed_mask",
            "target_has_bucket", "bucket_valid",
        ):
            flat[k] = flatten(stack_t(k, rollout_records))
        flat["old_log_prob"] = flat.pop("log_prob")
        flat["advantages"] = flatten(adv)
        flat["returns"] = flatten(ret)

        n_rows = flat["advantages"].shape[0]

        # ------- PPO update (Spinning Up style) -------
        t_train = time.perf_counter()
        metrics_accum = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
        }
        
        # 1. Policy Training with Early Stopping
        pi_steps = 0
        for _ in range(cfg.train_pi_iters):
            perm = np.random.permutation(n_rows)
            for start in range(0, n_rows, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb = {k: v[idx] for k, v in flat.items()}
                params, opt_state, _loss, m = policy_update(params, opt_state, mb)
                
                metrics_accum["policy_loss"] += float(m["policy_loss"])
                metrics_accum["entropy"] += float(m["entropy"])
                metrics_accum["approx_kl"] += float(m["approx_kl"])
                metrics_accum["clip_fraction"] += float(m["clip_fraction"])
                pi_steps += 1
            
            # Check for early stopping on CPU after each epoch
            if metrics_accum["approx_kl"] / max(pi_steps, 1) > 1.5 * cfg.target_kl:
                break
        
        if pi_steps > 0:
            for k in ("policy_loss", "entropy", "approx_kl", "clip_fraction"):
                metrics_accum[k] /= pi_steps

        # 2. Value Function Fitting
        v_steps = 0
        for _ in range(cfg.train_v_iters):
            perm = np.random.permutation(n_rows)
            for start in range(0, n_rows, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb = {k: v[idx] for k, v in flat.items()}
                params, opt_state, v_loss = value_update(params, opt_state, mb)
                metrics_accum["value_loss"] += float(v_loss)
                v_steps += 1
        
        if v_steps > 0:
            metrics_accum["value_loss"] /= v_steps

        train_s = time.perf_counter() - t_train

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
            # Get current LR values from schedules
            cur_step = update_idx * steps_per_update
            plr = float(pi_sched(cur_step))
            vlr = float(vf_sched(cur_step))

            log_print(
                f"{update_idx:6d} | {active_mode:8s} | "
                f"{learner_wr:7.1%} | "
                f"{wld:>5s} | "
                f"{episodes:7d} | {mean_ret:+.3f} | {env_sps:7.0f} | "
                f"{metrics_accum['policy_loss']:+.4f} | "
                f"{metrics_accum['value_loss']:.4f} | {metrics_accum['entropy']:.3f} | "
                f"{ev:+.3f} | {metrics_accum['approx_kl']:.5f} | {metrics_accum['clip_fraction']:.3f} | "
                f"{plr:.2e} | {vlr:.2e}"
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
            and len(display_window) >= cfg.heuristic_window_episodes
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
