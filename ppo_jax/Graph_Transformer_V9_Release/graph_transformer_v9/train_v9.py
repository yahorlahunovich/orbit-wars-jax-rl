"""Train GraphTransformerV9 using the shared JAX PPO backend."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.graph_transformer_v9.config import DEFAULT_CONFIG
from models.graph_transformer_v9.network_jax import GraphTransformerV9
from models.graph_transformer_v8.train_v8 import extract_obs_v8_jax, extract_scripted_obs_v8_jax
from models.planet_transformer_jax import train_jax as backend


MAX_PPO_LAUNCH_SLOTS = 16


def _normalized_entropy(probs, log_probs, valid_mask):
    """Return entropy normalized by the valid categorical support size."""
    raw_ent = -jnp.sum(probs * log_probs, axis=-1)
    valid_count = jnp.sum(valid_mask.astype(jnp.float32), axis=-1)
    denom = jnp.where(valid_count > 1.0, jnp.log(valid_count), 1.0)
    return jnp.where(valid_count > 1.0, raw_ent / denom, 0.0)


def _weighted_launch_entropy(tgt_probs, tgt_log_probs, tgt_mask, frac_probs, frac_log_probs, frac_mask):
    target_weight = float(DEFAULT_CONFIG.get("target_entropy_weight", 1.0))
    frac_weight = float(DEFAULT_CONFIG.get("frac_entropy_weight", 0.35))
    tgt_ent = _normalized_entropy(tgt_probs, tgt_log_probs, tgt_mask)
    frac_ent = _normalized_entropy(frac_probs, frac_log_probs, frac_mask)
    return target_weight * tgt_ent + frac_weight * frac_ent


def _send_logit_bias():
    return float(DEFAULT_CONFIG.get("send_logit_bias", 0.0))


def extract_obs_v9_jax(state, player_id) -> backend.ObsBatch:
    """V9 observation extraction with compact rollout storage dtypes."""
    obs = extract_obs_v8_jax(state, player_id)
    return backend.ObsBatch(
        node_features=obs.node_features,
        edge_features=obs.edge_features.astype(jnp.bfloat16),
        future_sight=obs.future_sight.astype(jnp.bfloat16),
        global_features=obs.global_features,
        owned_nodes=obs.owned_nodes,
        edge_valid_mask=obs.edge_valid_mask,
    )


def extract_scripted_obs_v9_jax(state, player_id) -> backend.ObsBatch:
    """V9-shaped scripted observation with compact storage dtypes."""
    obs = extract_scripted_obs_v8_jax(state, player_id)
    return backend.ObsBatch(
        node_features=obs.node_features,
        edge_features=obs.edge_features.astype(jnp.bfloat16),
        future_sight=obs.future_sight.astype(jnp.bfloat16),
        global_features=obs.global_features,
        owned_nodes=obs.owned_nodes,
        edge_valid_mask=obs.edge_valid_mask,
    )


def forward_and_sample_v9_compact(
    model: GraphTransformerV9,
    obs: backend.ObsBatch,
    rng: jnp.ndarray,
    temperature: jnp.ndarray = 1.0,
) -> tuple:
    """V9 rollout sampler that avoids no-launch target/fraction heads."""
    node_h, edge_h = model.encode(
        obs.node_features,
        obs.edge_features,
        obs.future_sight,
        obs.global_features,
    )
    head = model.policy_heads[0]

    M = obs.owned_nodes.shape[0]
    N = obs.node_features.shape[0]
    B = head.n_ship_options
    H = node_h.shape[-1]
    E = edge_h.shape[-1]

    src_safe = jnp.where(obs.owned_nodes >= 0, obs.owned_nodes, 0)
    slot_valid = obs.owned_nodes >= 0
    src_h = node_h[src_safe]
    send_cat = jnp.concatenate(
        [src_h, jnp.broadcast_to(obs.global_features[None, :], (M, 8))],
        axis=-1,
    )
    send_logits = (
        jax.nn.silu(send_cat @ head.w_send1.T + head.b_send1) @ head.w_send2.T + head.b_send2
    ).squeeze(-1) + _send_logit_bias()
    send_logits_scaled = send_logits / jnp.maximum(temperature, 1e-4)

    tgt_valid = obs.edge_valid_mask.any(axis=-1)
    target_ids = jnp.arange(N)[None, :]
    non_noop_target = target_ids != src_safe[:, None]
    combined_tgt_mask = tgt_valid & slot_valid[:, None] & non_noop_target
    slot_launch_feasible = slot_valid & combined_tgt_mask.any(axis=-1)

    rng_send, rng_tgt, rng_frac = jax.random.split(rng, 3)
    send_pair_logits = jnp.stack([jnp.zeros_like(send_logits_scaled), send_logits_scaled], axis=-1)
    send_pair_logits = jnp.where(
        slot_launch_feasible[:, None] | (jnp.arange(2)[None, :] == 0),
        send_pair_logits,
        -1e9,
    )
    raw_send = jax.vmap(lambda logits, r: jax.random.categorical(r, logits))(
        send_pair_logits, jax.random.split(rng_send, M)
    ).astype(jnp.int32)

    raw_launch = (raw_send == 1) & slot_launch_feasible
    launch_rank = jnp.where(raw_launch, jnp.arange(M), M + jnp.arange(M))
    launch_slots = jnp.sort(launch_rank)[:MAX_PPO_LAUNCH_SLOTS]
    launch_present = launch_slots < M
    safe_slots = jnp.where(launch_present, launch_slots, 0)
    selected_launch = (
        jnp.zeros((M,), dtype=jnp.int32)
        .at[safe_slots]
        .add(launch_present.astype(jnp.int32))
    ) > 0
    action_send = selected_launch.astype(jnp.int32)

    src_nodes = src_safe[safe_slots]
    src_h_k = node_h[src_nodes]
    edge_s = edge_h[src_nodes]
    raw_edge_s = obs.edge_features[src_nodes]
    pair_cat = jnp.concatenate(
        [
            jnp.broadcast_to(src_h_k[:, None, :], (MAX_PPO_LAUNCH_SLOTS, N, H)),
            jnp.broadcast_to(node_h[None, :, :], (MAX_PPO_LAUNCH_SLOTS, N, H)),
            edge_s,
            jnp.broadcast_to(obs.global_features[None, None, :], (MAX_PPO_LAUNCH_SLOTS, N, 8)),
        ],
        axis=-1,
    )
    flat_pair = pair_cat.reshape(-1, 2 * H + E + 8)
    target_base = (
        jax.nn.silu(flat_pair @ head.w_tgt1.T + head.b_tgt1) @ head.w_tgt2.T + head.b_tgt2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, N)

    bucket_raw = head._bucket_raw(raw_edge_s)
    bucket_cat = jnp.concatenate(
        [
            jnp.broadcast_to(pair_cat[:, :, None, :], (MAX_PPO_LAUNCH_SLOTS, N, B, 2 * H + E + 8)),
            bucket_raw,
        ],
        axis=-1,
    )
    flat_bucket = bucket_cat.reshape(-1, 2 * H + E + 8 + 4)
    bucket_utility = (
        jax.nn.silu(flat_bucket @ head.w_bucket1.T + head.b_bucket1) @ head.w_bucket2.T + head.b_bucket2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, N, B)
    target_logits = (target_base + jnp.max(bucket_utility, axis=-1)) / jnp.maximum(temperature, 1e-4)
    masked_tgt_logits = jnp.where(combined_tgt_mask[safe_slots] & launch_present[:, None], target_logits, -1e9)
    safe_tgt_logits = jnp.where(
        launch_present[:, None],
        masked_tgt_logits,
        jnp.where(jnp.arange(N)[None, :] == 0, 0.0, -1e9),
    )
    action_tgt_k = jax.vmap(lambda logits, r: jax.random.categorical(r, logits))(
        safe_tgt_logits, jax.random.split(rng_tgt, MAX_PPO_LAUNCH_SLOTS)
    )

    pair_chosen = pair_cat[jnp.arange(MAX_PPO_LAUNCH_SLOTS), action_tgt_k]
    bucket_raw_chosen = bucket_raw[jnp.arange(MAX_PPO_LAUNCH_SLOTS), action_tgt_k]
    frac_cat = jnp.concatenate(
        [
            jnp.broadcast_to(pair_chosen[:, None, :], (MAX_PPO_LAUNCH_SLOTS, B, 2 * H + E + 8)),
            bucket_raw_chosen,
        ],
        axis=-1,
    )
    flat_frac = frac_cat.reshape(-1, 2 * H + E + 8 + 4)
    frac_logits = (
        jax.nn.silu(flat_frac @ head.w_frac1.T + head.b_frac1) @ head.w_frac2.T + head.b_frac2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, B)
    frac_logits = frac_logits / jnp.maximum(temperature, 1e-4)
    frac_valid = obs.edge_valid_mask[safe_slots, action_tgt_k]
    masked_frac_logits = jnp.where(frac_valid & launch_present[:, None], frac_logits, -1e9)
    safe_frac_logits = jnp.where(
        launch_present[:, None],
        masked_frac_logits,
        jnp.where(jnp.arange(B)[None, :] == 0, 0.0, -1e9),
    )
    action_frac_k = jax.vmap(lambda logits, r: jax.random.categorical(r, logits))(
        safe_frac_logits, jax.random.split(rng_frac, MAX_PPO_LAUNCH_SLOTS)
    )

    encoded_tgt = (
        jnp.zeros((M,), dtype=jnp.int32)
        .at[safe_slots]
        .add(jnp.where(launch_present, action_tgt_k + 1, 0))
    )
    encoded_frac = (
        jnp.zeros((M,), dtype=jnp.int32)
        .at[safe_slots]
        .add(jnp.where(launch_present, action_frac_k + 1, 0))
    )
    action_tgt = jnp.where(encoded_tgt > 0, encoded_tgt - 1, -1)
    action_frac = jnp.where(encoded_frac > 0, encoded_frac - 1, 0)

    send_log_probs = jax.nn.log_softmax(send_pair_logits, axis=-1)
    send_probs = jax.nn.softmax(send_pair_logits, axis=-1)
    chosen_send_lp = send_log_probs[jnp.arange(M), action_send]
    tgt_log_probs = jax.nn.log_softmax(masked_tgt_logits, axis=-1)
    frac_log_probs = jax.nn.log_softmax(masked_frac_logits, axis=-1)
    chosen_tgt_lp_k = tgt_log_probs[jnp.arange(MAX_PPO_LAUNCH_SLOTS), action_tgt_k]
    chosen_frac_lp_k = frac_log_probs[jnp.arange(MAX_PPO_LAUNCH_SLOTS), action_frac_k]
    launched_extra_lp = jnp.zeros((M,), dtype=jnp.float32).at[safe_slots].add(
        jnp.where(launch_present, chosen_tgt_lp_k + chosen_frac_lp_k, 0.0)
    )

    num_owned = jnp.maximum(jnp.sum(slot_valid), 1.0)
    step_lp = jnp.sum(jnp.where(slot_valid, chosen_send_lp + launched_extra_lp, 0.0)) / num_owned

    tgt_probs = jax.nn.softmax(masked_tgt_logits, axis=-1)
    frac_probs = jax.nn.softmax(masked_frac_logits, axis=-1)
    launch_ent_k = _weighted_launch_entropy(
        tgt_probs,
        tgt_log_probs,
        combined_tgt_mask[safe_slots] & launch_present[:, None],
        frac_probs,
        frac_log_probs,
        frac_valid & launch_present[:, None],
    )
    launch_ent = jnp.zeros((M,), dtype=jnp.float32).at[safe_slots].add(
        jnp.where(launch_present, launch_ent_k, 0.0)
    )
    # Do not entropy-regularize the binary hold/launch head. Rewarding
    # uncertainty there directly increases APM. Entropy is only for target/bin
    # diversity conditional on launches that were actually sampled.
    entropy = jnp.sum(jnp.where(slot_valid, launch_ent, 0.0)) / num_owned

    value = model.value_head(obs.node_features, obs.future_sight, obs.global_features, obs.edge_features)
    action_tgt = jnp.where(slot_valid & (action_send == 1), action_tgt, -1)
    action_frac = jnp.where(slot_valid & (action_send == 1), action_frac, 0)
    return action_tgt, action_frac, step_lp, value.squeeze(), entropy


def compute_log_prob_v9_compact(
    model: GraphTransformerV9,
    obs: backend.ObsBatch,
    action_tgt: jnp.ndarray,
    action_frac: jnp.ndarray,
    temperature: jnp.ndarray = 1.0,
    apm_expected_temperature: jnp.ndarray = 1.0,
) -> tuple:
    """Exact PPO log-prob for taken actions without all-slot pair heads.

    PPO only needs target/fraction probabilities for source slots that actually
    launched. No-launch slots contribute only the binary send/no-send term.
    This keeps the V9 trunk and edge encoder trainable while avoiding the
    expensive 60x60xbucket head backward pass for inactive slots.
    """
    node_h, edge_h = model.encode(
        obs.node_features,
        obs.edge_features,
        obs.future_sight,
        obs.global_features,
    )
    head = model.policy_heads[0]

    M = obs.owned_nodes.shape[0]
    N = obs.node_features.shape[0]
    B = head.n_ship_options
    H = node_h.shape[-1]
    E = edge_h.shape[-1]

    src_safe = jnp.where(obs.owned_nodes >= 0, obs.owned_nodes, 0)
    slot_valid = obs.owned_nodes >= 0
    src_h = node_h[src_safe]
    send_cat = jnp.concatenate(
        [src_h, jnp.broadcast_to(obs.global_features[None, :], (M, 8))],
        axis=-1,
    )
    send_logits = (
        jax.nn.silu(send_cat @ head.w_send1.T + head.b_send1) @ head.w_send2.T + head.b_send2
    ).squeeze(-1) + _send_logit_bias()
    send_logits_scaled = send_logits / jnp.maximum(temperature, 1e-4)

    tgt_valid = obs.edge_valid_mask.any(axis=-1)
    target_ids = jnp.arange(N)[None, :]
    non_noop_target = target_ids != src_safe[:, None]
    combined_tgt_mask = tgt_valid & slot_valid[:, None] & non_noop_target
    slot_launch_feasible = slot_valid & combined_tgt_mask.any(axis=-1)

    raw_action_send = action_tgt >= 0
    action_send = raw_action_send & slot_launch_feasible
    send_pair_logits = jnp.stack([jnp.zeros_like(send_logits_scaled), send_logits_scaled], axis=-1)
    send_pair_logits = jnp.where(
        slot_launch_feasible[:, None] | (jnp.arange(2)[None, :] == 0),
        send_pair_logits,
        -1e9,
    )
    send_log_probs = jax.nn.log_softmax(send_pair_logits, axis=-1)
    send_probs = jax.nn.softmax(send_pair_logits, axis=-1)
    chosen_send_lp = send_log_probs[jnp.arange(M), action_send.astype(jnp.int32)]

    launch_rank = jnp.where(action_send, jnp.arange(M), M + jnp.arange(M))
    launch_slots = jnp.sort(launch_rank)[:MAX_PPO_LAUNCH_SLOTS]
    launch_present = launch_slots < M
    safe_slots = jnp.where(launch_present, launch_slots, 0)
    safe_tgt = jnp.clip(action_tgt[safe_slots], 0, N - 1)
    safe_frac = jnp.clip(action_frac[safe_slots], 0, B - 1)

    src_nodes = src_safe[safe_slots]
    src_h_k = node_h[src_nodes]
    edge_s = edge_h[src_nodes]
    raw_edge_s = obs.edge_features[src_nodes]

    pair_cat = jnp.concatenate(
        [
            jnp.broadcast_to(src_h_k[:, None, :], (MAX_PPO_LAUNCH_SLOTS, N, H)),
            jnp.broadcast_to(node_h[None, :, :], (MAX_PPO_LAUNCH_SLOTS, N, H)),
            edge_s,
            jnp.broadcast_to(obs.global_features[None, None, :], (MAX_PPO_LAUNCH_SLOTS, N, 8)),
        ],
        axis=-1,
    )
    flat_pair = pair_cat.reshape(-1, 2 * H + E + 8)
    target_base = (
        jax.nn.silu(flat_pair @ head.w_tgt1.T + head.b_tgt1) @ head.w_tgt2.T + head.b_tgt2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, N)

    bucket_raw = head._bucket_raw(raw_edge_s)
    bucket_cat = jnp.concatenate(
        [
            jnp.broadcast_to(pair_cat[:, :, None, :], (MAX_PPO_LAUNCH_SLOTS, N, B, 2 * H + E + 8)),
            bucket_raw,
        ],
        axis=-1,
    )
    flat_bucket = bucket_cat.reshape(-1, 2 * H + E + 8 + 4)
    bucket_utility = (
        jax.nn.silu(flat_bucket @ head.w_bucket1.T + head.b_bucket1) @ head.w_bucket2.T + head.b_bucket2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, N, B)
    target_logits = (target_base + jnp.max(bucket_utility, axis=-1)) / jnp.maximum(temperature, 1e-4)

    target_mask = combined_tgt_mask[safe_slots]
    masked_tgt_logits = jnp.where(target_mask & launch_present[:, None], target_logits, -1e9)
    tgt_log_probs = jax.nn.log_softmax(masked_tgt_logits, axis=-1)
    chosen_tgt_lp_k = tgt_log_probs[jnp.arange(MAX_PPO_LAUNCH_SLOTS), safe_tgt]

    pair_chosen = pair_cat[jnp.arange(MAX_PPO_LAUNCH_SLOTS), safe_tgt]
    bucket_raw_chosen = bucket_raw[jnp.arange(MAX_PPO_LAUNCH_SLOTS), safe_tgt]
    frac_cat = jnp.concatenate(
        [
            jnp.broadcast_to(pair_chosen[:, None, :], (MAX_PPO_LAUNCH_SLOTS, B, 2 * H + E + 8)),
            bucket_raw_chosen,
        ],
        axis=-1,
    )
    flat_frac = frac_cat.reshape(-1, 2 * H + E + 8 + 4)
    frac_logits = (
        jax.nn.silu(flat_frac @ head.w_frac1.T + head.b_frac1) @ head.w_frac2.T + head.b_frac2
    ).reshape(MAX_PPO_LAUNCH_SLOTS, B)
    frac_logits = frac_logits / jnp.maximum(temperature, 1e-4)
    frac_valid = obs.edge_valid_mask[safe_slots, safe_tgt]
    masked_frac_logits = jnp.where(frac_valid & launch_present[:, None], frac_logits, -1e9)
    frac_log_probs = jax.nn.log_softmax(masked_frac_logits, axis=-1)
    chosen_frac_lp_k = frac_log_probs[jnp.arange(MAX_PPO_LAUNCH_SLOTS), safe_frac]

    launched_extra_lp = jnp.zeros((M,), dtype=jnp.float32).at[safe_slots].add(
        jnp.where(launch_present, chosen_tgt_lp_k + chosen_frac_lp_k, 0.0)
    )
    num_owned = jnp.maximum(jnp.sum(slot_valid), 1.0)
    step_lp = jnp.sum(jnp.where(slot_valid, chosen_send_lp + launched_extra_lp, 0.0)) / num_owned

    p_launch_policy = jnp.where(slot_launch_feasible, send_probs[:, 1], 0.0)

    apm_send_logits_scaled = send_logits / jnp.maximum(apm_expected_temperature, 1e-4)
    apm_send_pair_logits = jnp.stack(
        [jnp.zeros_like(apm_send_logits_scaled), apm_send_logits_scaled],
        axis=-1,
    )
    apm_send_pair_logits = jnp.where(
        slot_launch_feasible[:, None] | (jnp.arange(2)[None, :] == 0),
        apm_send_pair_logits,
        -1e9,
    )
    apm_send_probs = jax.nn.softmax(apm_send_pair_logits, axis=-1)
    p_launch_apm = jnp.where(slot_launch_feasible, apm_send_probs[:, 1], 0.0)
    expected_launches = jnp.sum(p_launch_apm)
    feasible_launch_slots = jnp.sum(slot_launch_feasible.astype(jnp.float32))
    policy_count_dist = backend.poisson_binomial_count_dist(p_launch_policy)

    frac_probs = jax.nn.softmax(masked_frac_logits, axis=-1)
    tgt_probs = jax.nn.softmax(masked_tgt_logits, axis=-1)
    launch_ent_k = _weighted_launch_entropy(
        tgt_probs,
        tgt_log_probs,
        target_mask & launch_present[:, None],
        frac_probs,
        frac_log_probs,
        frac_valid & launch_present[:, None],
    )
    launch_ent = jnp.zeros((M,), dtype=jnp.float32).at[safe_slots].add(
        jnp.where(launch_present, launch_ent_k, 0.0)
    )
    # Match rollout sampling: entropy affects target/bin exploration, not the
    # binary hold/launch decision.
    entropy = jnp.sum(jnp.where(slot_valid, launch_ent, 0.0)) / num_owned

    value = model.value_head(obs.node_features, obs.future_sight, obs.global_features, obs.edge_features)
    prior_mu_log, prior_sigma_log = model.launch_prior_head(
        obs.node_features, obs.future_sight, obs.global_features
    )
    return (
        step_lp,
        value.squeeze(),
        entropy,
        expected_launches,
        policy_count_dist,
        feasible_launch_slots,
        prior_mu_log,
        prior_sigma_log,
    )


def _patch_backend() -> None:
    backend.PlanetEdgeTransformerJax = GraphTransformerV9
    backend.extract_obs_jax = extract_obs_v9_jax
    backend.extract_scripted_obs_jax = extract_scripted_obs_v9_jax
    backend.forward_and_sample = forward_and_sample_v9_compact
    backend.compute_log_prob = compute_log_prob_v9_compact



def train_main(cfg: dict):
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    for key in (
        "send_logit_bias",
        "target_entropy_weight",
        "frac_entropy_weight",
    ):
        if key in merged:
            DEFAULT_CONFIG[key] = merged[key]
    _patch_backend()
    if "batch_size" in merged:
        merged["n_envs"] = merged["batch_size"]

    checkpoint_dir = merged.get("checkpoint_dir")
    if checkpoint_dir is None and merged.get("output_root"):
        run_suffix = merged.get("_run_suffix", "")
        checkpoint_dir = (
            f"{merged['output_root']}/{run_suffix}" if run_suffix else merged["output_root"]
        )

    trainer = backend.JaxTrainer(merged, checkpoint_dir=checkpoint_dir)
    if merged.get("resume"):
        trainer.load_checkpoint(merged["resume"])
    trainer.train(total_steps=merged["total_steps"])


def main():
    _patch_backend()
    backend.main()


if __name__ == "__main__":
    main()
