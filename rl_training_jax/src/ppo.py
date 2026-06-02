"""PPO loss + GAE for the Transformer Orbit Wars policy.

All operations consume the per-source decision rows produced by
`orbit_wars.rollout.policy_step`. A "row" = one (env, time, source_planet)
triple. Rows where `source_valid == False` are masked out throughout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from orbit_wars.decode import BUCKET_COUNT, compose_action_grid


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,           # (B, T)
    values: jnp.ndarray,             # (B, T)
    dones: jnp.ndarray,              # (B, T) — True if episode ended at this step
    next_value: jnp.ndarray,         # (B,) — value bootstrap after last step
    gamma: float,
    lam: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation per env.

    Returns `(advantages, returns)` each shaped (B, T).

    Episode resets: when `dones[b, t] == True`, the bootstrap is zeroed at
    that step so advantages don't leak across episode boundaries.
    """
    not_done = 1.0 - dones.astype(jnp.float32)

    def scan_body(carry, x):
        next_v, next_gae = carry
        value, reward, nd = x
        delta = reward + gamma * next_v * nd - value
        gae = delta + gamma * lam * nd * next_gae
        return (value, gae), gae

    # Iterate in reverse over time (axis=1).
    rewards_t = jnp.transpose(rewards, (1, 0))         # (T, B)
    values_t = jnp.transpose(values, (1, 0))
    nd_t = jnp.transpose(not_done, (1, 0))

    init = (next_value, jnp.zeros_like(next_value))
    (_v, _gae), advs = jax.lax.scan(
        scan_body, init, (values_t, rewards_t, nd_t), reverse=True
    )
    advs = jnp.transpose(advs, (1, 0))                  # (B, T)
    returns = advs + values
    return advs, returns


# ---------------------------------------------------------------------------
# Joint log-prob (target + bucket) under a frozen action grid
# ---------------------------------------------------------------------------


_NEG_INF = jnp.float32(-1e9)


def _masked_log_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    safe = jnp.where(mask, logits, _NEG_INF)
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe = jnp.where(any_valid, safe, jnp.zeros_like(logits))
    return jax.nn.log_softmax(safe, axis=-1)


def joint_log_prob_and_entropy(
    target_logits: jnp.ndarray,         # (N, P, P)
    bucket_logits: jnp.ndarray,         # (N, P, P, BUCKETS)
    target_has_bucket: jnp.ndarray,     # (N, P, P) bool
    bucket_valid: jnp.ndarray,          # (N, P, P, BUCKETS) bool
    target_idx: jnp.ndarray,            # (N, P) int32
    bucket_idx: jnp.ndarray,            # (N, P) int32
    executed_mask: jnp.ndarray,         # (N, P) bool
) -> dict[str, jnp.ndarray]:
    """Return joint log_prob, per-row entropy contributions, all masked by
    `executed_mask` so PPO can flatten and average."""
    tgt_lp = _masked_log_softmax(target_logits, target_has_bucket)        # (N, P, P)
    tgt_lp_sel = jnp.take_along_axis(tgt_lp, target_idx[..., None], axis=-1).squeeze(-1)
    tgt_lp_sel = jnp.where(executed_mask, tgt_lp_sel, 0.0)

    # Per-source entropy of the target distribution (sum over targets).
    tgt_p = jnp.exp(tgt_lp) * target_has_bucket.astype(tgt_lp.dtype)
    entropy_target = -jnp.sum(tgt_p * tgt_lp, axis=-1)                    # (N, P)
    entropy_target = jnp.where(executed_mask, entropy_target, 0.0)

    # Gather bucket logits and validity for the *chosen* target (Point 3).
    b_idx = jnp.arange(target_idx.shape[0])[:, None]
    p_idx = jnp.arange(target_idx.shape[1])[None, :]
    chosen_bucket_logits = bucket_logits[b_idx, p_idx, target_idx]

    chosen_bucket_valid = jnp.take_along_axis(
        bucket_valid,
        target_idx[..., None, None].repeat(BUCKET_COUNT, axis=-1),
        axis=2,
    ).squeeze(2)                                                          # (N, P, BUCKETS)
    bkt_lp = _masked_log_softmax(chosen_bucket_logits, chosen_bucket_valid)
    bkt_lp_sel = jnp.take_along_axis(bkt_lp, bucket_idx[..., None], axis=-1).squeeze(-1)
    bkt_lp_sel = jnp.where(executed_mask, bkt_lp_sel, 0.0)

    bkt_p = jnp.exp(bkt_lp) * chosen_bucket_valid.astype(bkt_lp.dtype)
    entropy_bucket = -jnp.sum(bkt_p * bkt_lp, axis=-1)
    entropy_bucket = jnp.where(executed_mask, entropy_bucket, 0.0)

    return {
        "log_prob": tgt_lp_sel + bkt_lp_sel,
        "entropy_target": entropy_target,
        "entropy_bucket": entropy_bucket,
    }


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss_fn(
    params,
    apply_fn,
    batch: dict,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jnp.ndarray, dict]:
    """Compute PPO clipped-objective loss + diagnostic metrics.

    Shapes:
        N = batch rows (env, time) flattened
        P = MAX_PLANETS source decisions per row

    `batch` must contain:
        planet_features (N, P, F_p), planet_mask (N, P),
        fleet_features  (N, F, F_f), fleet_mask  (N, F),
        global_features (N, F_g)
            — encoder outputs (recomputed at minibatch time).
        target_idx, bucket_idx, executed_mask, old_log_prob   (N, P)
            — chosen actions and per-source validity from rollout time.
        target_has_bucket (N, P, P) bool, bucket_valid (N, P, P, BUCKETS) bool
            — frozen action-grid masks captured at rollout time.
        advantages (N,), returns (N,)
            — GAE outputs at env-time granularity; broadcast over sources.
    """
    out = apply_fn(
        params,
        planet_features=batch["planet_features"],
        planet_mask=batch["planet_mask"],
    )                                                # value (N,), target_logits (N,P,P), bucket_logits (N,P,P,B)

    info = joint_log_prob_and_entropy(
        target_logits=out.target_logits,
        bucket_logits=out.bucket_logits,
        target_has_bucket=batch["target_has_bucket"],
        bucket_valid=batch["bucket_valid"],
        target_idx=batch["target_idx"],
        bucket_idx=batch["bucket_idx"],
        executed_mask=batch["executed_mask"],
    )
    new_log_prob = info["log_prob"]                  # (N, P)
    entropy = info["entropy_target"] + info["entropy_bucket"]   # (N, P)

    old_log_prob = batch["old_log_prob"]             # (N, P)
    adv_env = batch["advantages"]                    # (N,)
    
    # Normalizing advantages per minibatch ensures consistent gradient magnitude.
    # This is critical for preventing 'timid' updates and very low KL.
    adv_mean = jnp.mean(adv_env)
    adv_std = jnp.std(adv_env)
    adv_norm = (adv_env - adv_mean) / jnp.maximum(adv_std, 1e-8)
    
    returns_env = batch["returns"]                   # (N,)
    executed_mask = batch["executed_mask"]           # (N, P)
    mask_f = executed_mask.astype(jnp.float32)
    mask_count = jnp.maximum(jnp.sum(mask_f), 1.0)

    adv = adv_norm[:, None]                          # (N, 1) — broadcast normalized across sources
    ratio = jnp.exp(new_log_prob - old_log_prob)
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
    policy_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * mask_f) / mask_count

    # Critic: one scalar per env-time row.
    value_pred = out.value                           # (N,)
    value_loss = jnp.mean((returns_env - value_pred) ** 2)

    entropy_mean = jnp.sum(entropy * mask_f) / mask_count

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

    log_ratio = new_log_prob - old_log_prob
    approx_kl = jnp.sum((ratio - 1.0 - log_ratio) * mask_f) / mask_count
    clip_frac = jnp.sum(((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32)) * mask_f) / mask_count

    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }


def explained_variance(returns: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """1 - Var(returns - values) / Var(returns). Returns 0 if returns is constant."""
    var_r = jnp.var(returns)
    return jnp.where(var_r < 1e-8, jnp.float32(0.0), 1.0 - jnp.var(returns - values) / var_r)
