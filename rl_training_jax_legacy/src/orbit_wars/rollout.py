"""Sample masked actions from the Transformer policy and pack them for `step_jit`.

Wiring:

    state, params -> features (Phase 1)
                  -> policy out (Phase 2)
                  -> decode target grid (Phase 3a)
                  -> sample target (HERE)
                  -> decode bucket grid for chosen targets (Phase 3b)
                  -> sample bucket (HERE)
                  -> pack (M, 3) action tensor + mask (HERE)

This two-stage decoding avoids the O(P*P*B) bottleneck.
"""

from __future__ import annotations

import functools
import jax
import jax.numpy as jnp

from .constants import MAX_MOVES_PER_PLAYER, INTERCEPT_ITERATIONS, SUN_PATH_MARGIN
from .decode import compose_target_grid, pack_action_row
from .state import OrbitWarsState

_NEG_INF = jnp.float32(-1e9)


def _masked_log_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """log_softmax with -inf for masked entries."""
    any_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe_logits = jnp.where(mask, logits, _NEG_INF)
    # Replace fully-masked rows with zero logits to avoid -inf log_softmax NaN.
    safe_logits = jnp.where(any_valid, safe_logits, jnp.zeros_like(logits))
    return jax.nn.log_softmax(safe_logits, axis=-1)


def _entropy_from_log_probs(log_probs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Sum -p log p over masked entries (axis=-1)."""
    p = jnp.exp(log_probs) * mask.astype(log_probs.dtype)
    return -jnp.sum(p * log_probs, axis=-1)


def sample_actions(
    rng: jax.Array,
    target_logits: jnp.ndarray,    # (B, P, P)
    bucket_logits: jnp.ndarray,    # (B, P, P, BUCKETS)
    state: OrbitWarsState,         # (B,)
    phase1: dict,                  # vmapped output of compose_target_grid
    deterministic: bool = False,
    intercept_iterations: int = INTERCEPT_ITERATIONS,
    sun_path_margin: float = SUN_PATH_MARGIN,
    **kwargs,
) -> dict[str, jnp.ndarray]:
    """Sample (target, bucket) per source planet with split-phase grid."""
    target_mask = phase1["target_mask"]              # (B, P, P)
    source_valid_any = phase1["source_valid_any"]    # (B, P)

    # 1. Target distribution: log_softmax with mask.
    tgt_log_probs = _masked_log_softmax(target_logits, target_mask)
    entropy_target = _entropy_from_log_probs(tgt_log_probs, target_mask)

    # Sample target.
    b, p, _ = target_logits.shape
    rng, k_tgt = jax.random.split(rng)
    tgt_keys = jax.random.split(k_tgt, b * p).reshape(b, p, 2)

    def _sample_target(lp_row, key):
        if deterministic:
            return jnp.argmax(lp_row, axis=-1)
        return jax.random.categorical(key, lp_row, axis=-1)

    target_idx = jax.vmap(jax.vmap(_sample_target))(tgt_log_probs, tgt_keys).astype(jnp.int32)
    # Mask out invalid sources
    target_idx = jnp.where(source_valid_any, target_idx, jnp.int32(0))

    tgt_lp = jnp.take_along_axis(tgt_log_probs, target_idx[..., None], axis=-1).squeeze(-1)
    tgt_lp = jnp.where(source_valid_any, tgt_lp, jnp.float32(0.0))

    # 2. Phase 2: Compute buckets for CHOSEN targets only (O(P*B) instead of O(P*P*B))
    from .decode import compose_bucket_grid
    bucket_grid = jax.vmap(functools.partial(
        compose_bucket_grid, 
        intercept_iterations=intercept_iterations,
        sun_path_margin=sun_path_margin,
        **kwargs,
    ))(state, target_idx, phase1)
    
    chosen_bucket_valid = bucket_grid["bucket_valid"] # (B, P, BUCKETS)
    
    # Gather bucket logits for chosen target: (B, P, BUCKETS)
    bi = jnp.arange(b)[:, None]
    si = jnp.arange(p)[None, :]
    chosen_bucket_logits = bucket_logits[bi, si, target_idx]

    bkt_log_probs = _masked_log_softmax(chosen_bucket_logits, chosen_bucket_valid)
    entropy_bucket = _entropy_from_log_probs(bkt_log_probs, chosen_bucket_valid)

    # 3. Sample bucket.
    rng, k_bkt = jax.random.split(rng)
    bkt_keys = jax.random.split(k_bkt, b * p).reshape(b, p, 2)
    bucket_idx = jax.vmap(jax.vmap(_sample_target))(bkt_log_probs, bkt_keys).astype(jnp.int32)
    bucket_idx = jnp.where(source_valid_any, bucket_idx, jnp.int32(0))

    bkt_lp = jnp.take_along_axis(bkt_log_probs, bucket_idx[..., None], axis=-1).squeeze(-1)
    bkt_lp = jnp.where(source_valid_any, bkt_lp, jnp.float32(0.0))

    return {
        "target_idx": target_idx,
        "bucket_idx": bucket_idx,
        "log_prob": tgt_lp + bkt_lp,
        "entropy_target": entropy_target,
        "entropy_bucket": entropy_bucket,
        "source_valid": source_valid_any,
        "chosen_bucket_valid": chosen_bucket_valid,
        "angle": bucket_grid["angle"],
        "ship_counts": bucket_grid["ship_counts"],
    }


def pack_padded_actions(
    target_idx: jnp.ndarray,        # (B, P)
    bucket_idx: jnp.ndarray,        # (B, P)
    source_valid: jnp.ndarray,      # (B, P)
    from_ids: jnp.ndarray,          # (B, P)
    angle: jnp.ndarray,             # (B, P, B)  -- GATHERED for chosen target
    ship_counts: jnp.ndarray,       # (B, P, B)  -- GATHERED for chosen target
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Packs the chosen moves into a (B, 48, 3) action tensor."""
    b, p = target_idx.shape
    
    # Gather chosen angle and ships
    bi = jnp.arange(b)[:, None]
    si = jnp.arange(p)[None, :]
    chosen_angle = angle[bi, si, bucket_idx]
    chosen_ships = ship_counts[bi, si, bucket_idx]
    
    # Identify NOOP moves (target == source)
    s_range = jnp.arange(p)
    is_noop = (target_idx == s_range[None, :])
    env_mask = source_valid & (~is_noop)

    rows, mask_vals = jax.vmap(pack_action_row)(from_ids, chosen_angle, chosen_ships, env_mask)

    # Compact sources so all valid ones are first.
    sort_key = (-mask_vals).astype(jnp.float32)
    sort_idx = jnp.argsort(sort_key, axis=-1)
    rows_sorted = jnp.take_along_axis(rows, sort_idx[..., None].repeat(3, axis=-1), axis=1)
    mask_sorted = jnp.take_along_axis(mask_vals, sort_idx, axis=-1)

    actions = rows_sorted[:, :MAX_MOVES_PER_PLAYER, :]
    action_mask = mask_sorted[:, :MAX_MOVES_PER_PLAYER]

    # executed_mask for PPO. Includes NOOPs so they contribute to entropy and policy gradients.
    rank = jnp.argsort(sort_idx, axis=-1)
    executed_mask = source_valid & (rank < MAX_MOVES_PER_PLAYER)

    return actions, action_mask, executed_mask


def policy_step(
    rng: jax.Array,
    policy_apply,
    params,
    states,                          # vmapped OrbitWarsState
    features: dict,                  # vmapped encoder output
    player_per_env: jnp.ndarray,     # (B,) int32
    deterministic: bool = False,
) -> dict:
    """One full policy step with split-phase grid composition."""
    out = policy_apply(params, **features)
    
    # Phase 1: Target Grid
    phase1 = jax.vmap(compose_target_grid, in_axes=(0, 0, 0, 0))(
        states, player_per_env, features["incoming_me"], features["incoming_enemy"]
    )
    
    # Phase 2: Sample & Bucket Grid
    sampled = sample_actions(
        rng, out.target_logits, out.bucket_logits, states, phase1, deterministic=deterministic
    )
    
    # Phase 3: Pack
    actions, action_mask, executed_mask = pack_padded_actions(
        sampled["target_idx"], sampled["bucket_idx"], sampled["source_valid"],
        phase1["from_ids"], sampled["angle"], sampled["ship_counts"]
    )
    
    return {
        "actions": actions,
        "action_mask": action_mask,
        "target_idx": sampled["target_idx"],
        "bucket_idx": sampled["bucket_idx"],
        "log_prob": sampled["log_prob"],
        "entropy_target": sampled["entropy_target"],
        "entropy_bucket": sampled["entropy_bucket"],
        "executed_mask": executed_mask,
        "value": out.value,
        "target_has_bucket": phase1["target_mask"], # (B, P, P)
        "chosen_bucket_valid": sampled["chosen_bucket_valid"], # (B, P, B)
    }
