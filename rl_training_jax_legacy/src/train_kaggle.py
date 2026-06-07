"""JAX PPO training for Kaggle GPU notebooks."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from orbit_wars.convert import state_to_observation_dict
from orbit_wars.env import OrbitWarsJaxEnv
from policy import PlanetPolicy
from ppo import categorical_sample, ppo_loss

OBS_KEYS = (
    "self_features",
    "candidate_features",
    "global_features",
    "candidate_mask",
    "ship_bucket_mask",
    "bucket_features",
)


@dataclass(slots=True)
class StepGroup:
    indices: list[int]
    reward: float
    done: bool


@dataclass(slots=True)
class TrainConfig:
    seed: int = 321
    run_name: str = "jax_scratch_ppo"
    save_dir: str = "artifacts"
    episode_steps: int = 200
    candidate_count: int = 49
    ship_bucket_count: int = 5
    hidden_size: int = 128
    num_envs: int = 8
    rollout_steps: int = 32
    total_updates: int = 100
    epochs: int = 3
    minibatch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 1e-3
    lr_end: float = 1e-4
    max_grad_norm: float = 0.5
    log_every: int = 1
    checkpoint_every: int = 25
    opponent: str = "random"


def load_config(path: str | Path) -> TrainConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    env = data.get("env", {})
    model = data.get("model", {})
    ppo = data.get("ppo", {})
    return TrainConfig(
        seed=int(data.get("seed", 321)),
        run_name=str(data.get("run_name", "jax_scratch_ppo")),
        save_dir=str(data.get("save_dir", "artifacts")),
        episode_steps=int(env.get("episode_steps", 200)),
        candidate_count=int(env.get("candidate_count", 49)),
        ship_bucket_count=int(env.get("ship_bucket_count", 5)),
        hidden_size=int(model.get("hidden_size", 128)),
        num_envs=int(ppo.get("num_envs", 8)),
        rollout_steps=int(ppo.get("rollout_steps", 32)),
        total_updates=int(ppo.get("total_updates", 100)),
        epochs=int(ppo.get("epochs", 3)),
        minibatch_size=int(ppo.get("minibatch_size", 1024)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_coef=float(ppo.get("clip_coef", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        lr=float(ppo.get("lr", 1e-3)),
        lr_end=float(ppo.get("lr_end", 1e-4)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        log_every=int(data.get("log_every", 1)),
        checkpoint_every=int(data.get("checkpoint_every", 25)),
        opponent=str(data.get("opponent", "random")),
    )


def _ensure_rl_features_on_path() -> None:
    import sys
    from pathlib import Path

    work_roots = [
        Path("/kaggle/working"),
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    for work in work_roots:
        config_py = work / "rl_features" / "config.py"
        if config_py.is_file():
            root = str(work.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ModuleNotFoundError(
        "Could not find rl_features/config.py. On Kaggle, run the notebook cells that "
        "%%writefile rl_features/* before calling train()."
    )


def _features():
    _ensure_rl_features_on_path()
    from rl_features.config import EnvConfig
    from rl_features.features import (
        bucket_feature_dim,
        candidate_feature_dim,
        encode_turn,
        global_feature_dim,
        self_feature_dim,
    )

    return EnvConfig, encode_turn, self_feature_dim, candidate_feature_dim, global_feature_dim, bucket_feature_dim


def encode_obs(obs: dict[str, Any], cfg: TrainConfig, env_index: int):
    EnvConfig, encode_turn, *_ = _features()
    env_cfg = EnvConfig(
        episode_steps=cfg.episode_steps,
        candidate_count=cfg.candidate_count,
        ship_bucket_count=cfg.ship_bucket_count,
    )
    return encode_turn(obs, env_cfg, env_index=env_index)


def random_opponent(obs: dict[str, Any]) -> list[list[float | int]]:
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

    return list(
        random_agent(
            {
                "player": int(obs.get("player", 1)),
                "planets": list(obs.get("planets", [])),
            }
        )
    )


def build_moves(batch, tgt_idx: int, bucket_idx: int, context) -> list[float | int] | None:
    if not (
        tgt_idx > 0
        and tgt_idx < len(context.candidate_ids)
        and context.candidate_mask[tgt_idx]
        and context.ship_bucket_mask[tgt_idx, bucket_idx]
    ):
        return None
    ships = int(context.ship_count_buckets[tgt_idx][bucket_idx])
    if ships <= 0:
        return None
    for planet in batch.state.planets:
        if planet.id == context.source_id:
            if planet.ships < ships:
                return None
            return [context.source_id, float(context.target_angles[tgt_idx]), ships]
    return None


def policy_apply(model: PlanetPolicy, params, batch: dict[str, jnp.ndarray]):
    return model.apply(params, **{k: batch[k] for k in OBS_KEYS})


def _jax_int(x) -> int:
    arr = jnp.asarray(x)
    return int(arr.item() if arr.ndim == 0 else arr.reshape(-1)[0].item())


def _jax_float(x) -> float:
    arr = jnp.asarray(x)
    return float(arr.item() if arr.ndim == 0 else arr.reshape(-1)[0].item())


def merge_batches(batches) -> tuple[dict[str, jnp.ndarray] | None, int]:
    n_rows = sum(b.self_features.shape[0] for b in batches)
    if n_rows == 0:
        return None, 0
    return {
        "self_features": jnp.asarray(np.concatenate([b.self_features for b in batches], axis=0)),
        "candidate_features": jnp.asarray(np.concatenate([b.candidate_features for b in batches], axis=0)),
        "candidate_mask": jnp.asarray(np.concatenate([b.candidate_mask for b in batches], axis=0)),
        "ship_bucket_mask": jnp.asarray(np.concatenate([b.ship_bucket_mask for b in batches], axis=0)),
        "bucket_features": jnp.asarray(np.concatenate([b.bucket_features for b in batches], axis=0)),
    }, n_rows


def sample_policy_rows(rng, model, params, merged: dict[str, jnp.ndarray] | None, n_rows: int):
    empty_f = np.zeros((0,), np.float32)
    empty_i = np.zeros((0,), np.int32)
    if merged is None or n_rows == 0:
        return rng, empty_f, empty_i, empty_i, empty_f

    out = policy_apply(model, params, merged)
    targets, buckets, lps, vals = [], [], [], []
    for i in range(n_rows):
        rng, k1, k2 = jax.random.split(rng, 3)
        t_idx, t_lp, _ = categorical_sample(k1, out.target_logits[i])
        b_idx, b_lp, _ = categorical_sample(k2, out.ship_bucket_logits[i, t_idx])
        targets.append(_jax_int(t_idx))
        buckets.append(_jax_int(b_idx))
        lps.append(_jax_float(t_lp + b_lp))
        vals.append(_jax_float(out.value[i]))
    return (
        rng,
        np.asarray(vals, np.float32),
        np.asarray(targets, np.int32),
        np.asarray(buckets, np.int32),
        np.asarray(lps, np.float32),
    )


def collect_rollout(rng, model, params, envs, batches, cfg, next_seed):
    self_rows, cand_rows, glob_rows = [], [], []
    cmasks, smasks, bfeats = [], [], []
    target_indices, bucket_indices, log_probs, values = [], [], [], []
    groups_per_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards, step_rewards = [], []
    running_returns = [0.0] * len(envs)
    env_steps = 0

    for _ in range(cfg.rollout_steps):
        merged, n_rows = merge_batches(batches)
        offsets = np.cumsum([0] + [b.self_features.shape[0] for b in batches[:-1]])
        rng, row_values, tgt_idx_arr, bucket_idx_arr, lp_arr = sample_policy_rows(rng, model, params, merged, n_rows)

        next_batches = []
        for env_idx, (env, batch) in enumerate(zip(envs, batches)):
            start = int(offsets[env_idx])
            moves: list[list[float | int]] = []
            group_indices: list[int] = []

            for local_idx, context in enumerate(batch.contexts):
                g = start + local_idx
                self_rows.append(batch.self_features[local_idx])
                cand_rows.append(batch.candidate_features[local_idx])
                glob_rows.append(batch.global_features[local_idx])
                cmasks.append(batch.candidate_mask[local_idx])
                smasks.append(batch.ship_bucket_mask[local_idx])
                bfeats.append(batch.bucket_features[local_idx])
                values.append(float(row_values[g]) if n_rows else 0.0)
                t_idx = int(tgt_idx_arr[g]) if n_rows else 0
                b_idx = int(bucket_idx_arr[g]) if n_rows else 0
                target_indices.append(t_idx)
                bucket_indices.append(b_idx)
                log_probs.append(float(lp_arr[g]) if n_rows else 0.0)
                group_indices.append(len(values) - 1)
                move = build_moves(batch, t_idx, b_idx, context)
                if move is not None:
                    moves.append(move)

            opp_obs = state_to_observation_dict(env.state, player=env.learner_player ^ 1)
            opp_moves = random_opponent(opp_obs) if cfg.opponent == "random" else []
            result = env.step(moves, opp_moves)
            env_steps += 1
            reward = float(result.rewards[env.learner_player]) if result.done else 0.0
            step_rewards.append(reward)
            running_returns[env_idx] += reward
            groups_per_env[env_idx].append(StepGroup(group_indices, reward, result.done))

            if result.done:
                episode_rewards.append(running_returns[env_idx])
                running_returns[env_idx] = 0.0
                next_seed += 1
                next_batches.append(encode_obs(env.reset(seed=next_seed), cfg, env_idx))
            else:
                next_batches.append(encode_obs(result.observation, cfg, env_idx))
        batches = next_batches

    returns = [0.0] * len(values)
    advantages = [0.0] * len(values)
    merged_end, _ = merge_batches(batches)
    next_vals = [0.0] * len(envs)
    if merged_end is not None:
        out = policy_apply(model, params, merged_end)
        all_v = np.asarray(out.value)
        off = 0
        for i, batch in enumerate(batches):
            c = batch.self_features.shape[0]
            next_vals[i] = float(all_v[off : off + c].mean()) if c else 0.0
            off += c

    for env_idx, groups in enumerate(groups_per_env):
        last_gae = 0.0
        nxt = next_vals[env_idx]
        for group in reversed(groups):
            gv = float(np.mean([values[i] for i in group.indices])) if group.indices else 0.0
            nd = 1.0 - float(group.done)
            delta = group.reward + cfg.gamma * nxt * nd - gv
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nd * last_gae
            for idx in group.indices:
                advantages[idx] = last_gae
                returns[idx] = last_gae + values[idx]
            nxt = gv

    adv = np.asarray(advantages, dtype=np.float32)
    if adv.size:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    transition = {
        "self_features": jnp.asarray(np.asarray(self_rows, np.float32)),
        "candidate_features": jnp.asarray(np.asarray(cand_rows, np.float32)),
        "candidate_mask": jnp.asarray(np.asarray(cmasks, bool)),
        "ship_bucket_mask": jnp.asarray(np.asarray(smasks, bool)),
        "bucket_features": jnp.asarray(np.asarray(bfeats, np.float32)),
        "target_index": jnp.asarray(target_indices, dtype=jnp.int32),
        "ship_bucket_index": jnp.asarray(bucket_indices, dtype=jnp.int32),
        "old_log_prob": jnp.asarray(log_probs, dtype=jnp.float32),
        "returns": jnp.asarray(returns, dtype=jnp.float32),
        "advantages": jnp.asarray(adv, dtype=jnp.float32),
    }
    stats = {
        "env_steps": env_steps,
        "step_reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "episode_return_mean": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "episodes_finished": len(episode_rewards),
        "samples": len(values),
    }
    return transition, batches, next_seed, stats, rng


def explained_variance(returns: jnp.ndarray, values: jnp.ndarray) -> float:
    var_r = float(jnp.var(returns))
    if var_r < 1e-8:
        return float("nan")
    return 1.0 - float(jnp.var(returns - values) / var_r)


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    _, _, self_dim_fn, cand_dim_fn, glob_dim_fn, bucket_dim_fn = _features()
    self_dim = self_dim_fn()
    cand_dim = cand_dim_fn()
    glob_dim = glob_dim_fn()
    bucket_dim = bucket_dim_fn()

    model = PlanetPolicy(
        candidate_count=cfg.candidate_count,
        ship_bucket_count=cfg.ship_bucket_count,
        hidden_size=cfg.hidden_size,
        self_dim=self_dim,
        candidate_dim=cand_dim,
        global_dim=glob_dim,
        bucket_feature_dim=bucket_dim,
    )
    init_batch = {
        "self_features": jnp.zeros((2, self_dim), jnp.float32),
        "candidate_features": jnp.zeros((2, cfg.candidate_count, cand_dim), jnp.float32),
        "candidate_mask": jnp.ones((2, cfg.candidate_count), jnp.bool_),
        "ship_bucket_mask": jnp.ones((2, cfg.candidate_count, cfg.ship_bucket_count), jnp.bool_),
        "bucket_features": jnp.zeros((2, cfg.candidate_count, cfg.ship_bucket_count, bucket_dim), jnp.float32),
    }
    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, **init_batch)

    schedule = optax.linear_schedule(cfg.lr, cfg.lr_end, cfg.total_updates)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule))
    opt_state = optimizer.init(params)
    apply_fn = lambda p, **kw: model.apply(p, **kw)

    envs = [
        OrbitWarsJaxEnv(seed=cfg.seed + i, episode_steps=cfg.episode_steps, learner_player=i % 2)
        for i in range(cfg.num_envs)
    ]
    next_seed = cfg.seed + cfg.num_envs
    batches = [encode_obs(env.reset(seed=cfg.seed + i), cfg, i) for i, env in enumerate(envs)]

    save_dir = Path(cfg.save_dir) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"JAX devices: {jax.devices()} | envs={cfg.num_envs} rollout={cfg.rollout_steps} "
        f"updates={cfg.total_updates} env_steps/update={cfg.num_envs * cfg.rollout_steps}"
    )
    print(
        "update | upd/s | env_steps/s | rollout_s | train_s | ep_return | episodes | samples | "
        "loss | policy | value | entropy | ev | approx_kl | clip_frac"
    )

    t_start = time.perf_counter()
    total_env_steps = 0

    for update in range(1, cfg.total_updates + 1):
        t_rollout = time.perf_counter()
        transition, batches, next_seed, stats, rng = collect_rollout(
            rng, model, params, envs, batches, cfg, next_seed
        )
        rollout_s = time.perf_counter() - t_rollout
        total_env_steps += stats["env_steps"]

        n = int(transition["self_features"].shape[0])
        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        opt_steps = 0
        t_train = time.perf_counter()

        if n > 0:
            for _ in range(cfg.epochs):
                perm = np.random.permutation(n)
                for start in range(0, n, cfg.minibatch_size):
                    idx = perm[start : start + cfg.minibatch_size]
                    mb = {k: v[idx] for k, v in transition.items()}

                    def batch_loss(p):
                        return ppo_loss(p, apply_fn, mb, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef)

                    (loss, m), grads = jax.value_and_grad(batch_loss, has_aux=True)(params)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    metrics["loss"] += float(loss)
                    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
                        metrics[key] += float(m[key])
                    opt_steps += 1

        train_s = time.perf_counter() - t_train
        if opt_steps:
            for key in metrics:
                metrics[key] /= opt_steps

        ev = float("nan")
        if n > 0:
            out = policy_apply(model, params, {k: transition[k] for k in OBS_KEYS})
            ev = explained_variance(transition["returns"], out.value)

        elapsed = time.perf_counter() - t_start
        upd_s = update / elapsed
        env_sps = total_env_steps / elapsed
        ep = stats["episode_return_mean"]
        ep_s = "n/a" if np.isnan(ep) else f"{ep:+.3f}"

        if update % cfg.log_every == 0:
            ev_s = "n/a" if np.isnan(ev) else f"{ev:.3f}"
            print(
                f"{update:6d} | {upd_s:5.2f} | {env_sps:9.0f} | {rollout_s:7.1f} | {train_s:5.1f} | "
                f"{ep_s:>8} | {stats['episodes_finished']:8d} | {stats['samples']:7d} | "
                f"{metrics['loss']:.4f} | {metrics['policy_loss']:.4f} | {metrics['value_loss']:.4f} | "
                f"{metrics['entropy']:.3f} | {ev_s:>3} | {metrics['approx_kl']:.6f} | {metrics['clip_fraction']:.3f}"
            )

        if update % cfg.checkpoint_every == 0 or update == cfg.total_updates:
            blob = np.frombuffer(flax.serialization.to_bytes(params), dtype=np.uint8)
            np.savez(save_dir / "ckpt_last.npz", update=update, params=blob)
            np.savez(save_dir / f"ckpt_{update:06d}.npz", update=update, params=blob)

    print(f"Done. total_env_steps={total_env_steps} elapsed={time.perf_counter()-t_start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle_jax_train.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
