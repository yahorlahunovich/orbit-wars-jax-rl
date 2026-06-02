"""Sample masked actions from the Transformer policy and pack them for `step_jit`.

Wiring:

    state, params -> features (Phase 1)
                  -> policy out (Phase 2)
                  -> decode grid (Phase 3)
                  -> apply masks, sample target & bucket  (HERE)
                  -> pack (M, 3) action tensor + mask     (HERE)

Sampling is two-stage:

1. For every owned source planet, sample a target slot from the masked
   target logits (categorical with -inf on invalid targets).
2. Conditional on the chosen target, sample a bucket from masked bucket
   logits.

If a source planet has *no* valid (target, bucket) combination, its move is
masked out at the action-packing step.

We also compute the joint log-probability and per-row entropy needed by PPO.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import MAX_MOVES_PER_PLAYER, MAX_PLANETS
from .decode import BUCKET_COUNT, compose_action_grid

_NEG_INF = jnp.float32(-1e9)


def _masked_log_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """log_softmax with -inf for masked entries.

    If ALL entries are masked, we return a safe uniform log-prob so downstream
    sampling doesn't NaN; the caller still knows the source is invalid via the
    source-level mask.
    """
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
    bucket_logits: jnp.ndarray,    # (B, P, BUCKETS)
    action_grid: dict,             # output of compose_action_grid per env (vmapped: leading B)
    deterministic: bool = False,
) -> dict[str, jnp.ndarray]:
    """Sample (target, bucket) per source planet with full masking.

    Inputs are batched (`B` = num envs). `action_grid` should be the vmapped
    output of `compose_action_grid` (i.e. each value has a leading `B` axis).

    Returns dict with:
        target_idx       (B, P) int32         chosen target slot per source
        bucket_idx       (B, P) int32         chosen bucket per source
        log_prob         (B, P) float32       joint log-prob of (target, bucket)
        entropy_target   (B, P) float32       entropy of target distribution
        entropy_bucket   (B, P) float32       entropy of bucket dist (under chosen target)
        source_valid     (B, P) bool          source planet is owned AND has >=1 valid action
        target_valid_any (B, P) bool          at least one valid target for this source
    """
    pair_valid = action_grid["pair_valid"]          # (B, P, P)
    full_valid = action_grid["full_valid"]           # (B, P, P, BUCKETS)
    source_owned = action_grid["source_valid"]       # (B, P)

    # A target is choosable if any bucket is legal for that (source, target).
    target_has_bucket = jnp.any(full_valid, axis=-1)     # (B, P, P)
    target_valid_any = jnp.any(target_has_bucket, axis=-1)              # (B, P)
    source_valid = source_owned & target_valid_any                      # (B, P)

    # Target distribution: log_softmax with mask.
    tgt_log_probs = _masked_log_softmax(target_logits, target_has_bucket)
    entropy_target = _entropy_from_log_probs(tgt_log_probs, target_has_bucket)

    # Sample target. We split rng across (B, P) by folding indices.
    b, p, _ = target_logits.shape
    rng, k_tgt = jax.random.split(rng)
    tgt_keys = jax.random.split(k_tgt, b * p).reshape(b, p, 2)

    def _sample_target(lp_row, key):
        if deterministic:
            return jnp.argmax(lp_row, axis=-1)
        return jax.random.categorical(key, lp_row, axis=-1)

    target_idx = jax.vmap(jax.vmap(_sample_target))(tgt_log_probs, tgt_keys).astype(jnp.int32)
    # Force a safe value when source has no valid action (we'll mask the row
    # out at packing time anyway).
    target_idx = jnp.where(source_valid, target_idx, jnp.int32(0))

    tgt_lp = jnp.take_along_axis(tgt_log_probs, target_idx[..., None], axis=-1).squeeze(-1)
    tgt_lp = jnp.where(source_valid, tgt_lp, jnp.float32(0.0))

    # Bucket distribution conditional on the chosen target (Point 3).
    # Index into the (B, P, P, BUCKETS) bucket_logits using target_idx.
    b_idx = jnp.arange(b)[:, None]
    p_idx = jnp.arange(p)[None, :]
    chosen_bucket_logits = bucket_logits[b_idx, p_idx, target_idx] # (B, P, BUCKETS)

    # Gather full_valid[b, s, target_idx[b, s], :] -> (B, P, BUCKETS)
    chosen_bucket_valid = jnp.take_along_axis(
        full_valid, target_idx[..., None, None].repeat(BUCKET_COUNT, axis=-1), axis=2
    ).squeeze(2)                                                # (B, P, BUCKETS)
    bucket_lp_row = _masked_log_softmax(chosen_bucket_logits, chosen_bucket_valid)
    entropy_bucket = _entropy_from_log_probs(bucket_lp_row, chosen_bucket_valid)

    rng, k_bkt = jax.random.split(rng)
    bkt_keys = jax.random.split(k_bkt, b * p).reshape(b, p, 2)

    def _sample_bucket(lp_row, key):
        if deterministic:
            return jnp.argmax(lp_row, axis=-1)
        return jax.random.categorical(key, lp_row, axis=-1)

    bucket_idx = jax.vmap(jax.vmap(_sample_bucket))(bucket_lp_row, bkt_keys).astype(jnp.int32)
    bucket_idx = jnp.where(source_valid, bucket_idx, jnp.int32(0))

    bkt_lp = jnp.take_along_axis(bucket_lp_row, bucket_idx[..., None], axis=-1).squeeze(-1)
    bkt_lp = jnp.where(source_valid, bkt_lp, jnp.float32(0.0))

    log_prob = tgt_lp + bkt_lp

    return {
        "target_idx": target_idx,
        "bucket_idx": bucket_idx,
        "log_prob": log_prob,
        "entropy_target": entropy_target,
        "entropy_bucket": entropy_bucket,
        "source_valid": source_valid,
        "target_valid_any": target_valid_any,
    }


def pack_padded_actions(
    target_idx: jnp.ndarray,        # (B, P)
    bucket_idx: jnp.ndarray,        # (B, P)
    source_valid: jnp.ndarray,      # (B, P)
    action_grid: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pack per-source decisions into `(B, MAX_MOVES_PER_PLAYER, 3)` action tensors.

    Source planets without a valid action contribute mask=0 (zero row).
    Because MAX_PLANETS >= MAX_MOVES_PER_PLAYER could be violated (96 > 48),
    we *sort* sources by `source_valid` so all valid sources land in the
    first MAX_MOVES_PER_PLAYER slots, then truncate.
    """
    b, p = target_idx.shape
    from_ids = action_grid["from_ids"]                          # (B, P)
    angle_grid = action_grid["angle"]                           # (B, P, P, BUCKETS)
    ship_counts = action_grid["ship_counts"]                    # (B, P, P, BUCKETS)

    # Gather chosen angle/ship_count per source.
    s_range = jnp.arange(p, dtype=jnp.int32)
    b_range = jnp.arange(b, dtype=jnp.int32)
    bi, si = jnp.meshgrid(b_range, s_range, indexing="ij")
    angle = angle_grid[bi, si, target_idx, bucket_idx]          # (B, P)
    ships = ship_counts[bi, si, target_idx, bucket_idx]         # (B, P)

    # Build the per-source row.
    rows = jnp.stack([from_ids, angle, ships], axis=-1)         # (B, P, 3)

    # Identify NOOP moves (target == source)
    is_noop = (target_idx == s_range[None, :])                  # (B, P)

    # Mask is true only for valid sources that are NOT noops.
    # We zero out the action for NOOPs so no ships are launched.
    env_mask = source_valid & (~is_noop)
    mask = env_mask.astype(jnp.float32)                         # (B, P)
    rows = rows * mask[..., None]

    # Compact sources so all valid ones are first. We sort by NEGATIVE mask
    # (valid sources have key=-1 -> sort first), then truncate.
    sort_key = (-mask).astype(jnp.float32)
    sort_idx = jnp.argsort(sort_key, axis=-1)                   # (B, P)
    rows_sorted = jnp.take_along_axis(rows, sort_idx[..., None].repeat(3, axis=-1), axis=1)
    mask_sorted = jnp.take_along_axis(mask, sort_idx, axis=-1)

    actions = rows_sorted[:, :MAX_MOVES_PER_PLAYER, :]
    action_mask = mask_sorted[:, :MAX_MOVES_PER_PLAYER]

    # Truncation fix (Point 5): identify which sources actually "made the cut".
    # PPO should only learn from actions that were not truncated.
    rank = jnp.argsort(sort_idx, axis=-1)                       # (B, P) — rank of each source in sorted list
    executed_mask = source_valid & (rank < MAX_MOVES_PER_PLAYER) & (~is_noop)

    return actions, action_mask, executed_mask


def policy_step(
    rng: jax.Array,
    policy_apply,
    params,
    states,                          # vmapped (leading B) OrbitWarsState
    features: dict,                  # vmapped encoder output
    player_per_env: jnp.ndarray,     # (B,) int32 — which player this batch sees
    deterministic: bool = False,
) -> dict:
    """One full policy step:

    encode (done outside) -> policy forward -> sample masked actions -> pack.

    Returns a dict containing the action tensor + mask (ready for `step_jit`)
    and the per-row info PPO needs (`log_prob`, \`entropy\`, \`value\`,
    \`executed_mask\`).
    """
    out = policy_apply(params, **features)
    grid = jax.vmap(compose_action_grid, in_axes=(0, 0))(states, player_per_env)
    sampled = sample_actions(
        rng, out.target_logits, out.bucket_logits, grid, deterministic=deterministic
    )
    actions, action_mask, executed_mask = pack_padded_actions(
        sampled["target_idx"], sampled["bucket_idx"], sampled["source_valid"], grid
    )
    return {
        "actions": actions,                          # (B, MAX_MOVES, 3)
        "action_mask": action_mask,                  # (B, MAX_MOVES)
        "target_idx": sampled["target_idx"],         # (B, P) — for PPO
        "bucket_idx": sampled["bucket_idx"],
        "log_prob": sampled["log_prob"],
        "entropy_target": sampled["entropy_target"],
        "entropy_bucket": sampled["entropy_bucket"],
        "executed_mask": executed_mask,              # (B, P) — for PPO (fixes truncation bias)
        "value": out.value,                          # (B,)
    }
