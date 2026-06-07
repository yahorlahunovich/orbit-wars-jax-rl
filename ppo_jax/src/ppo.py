import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import NamedTuple, Any

from .policy import GraphTransformerV9
from .env import OrbitWarsPureJaxEnv, compute_reward
from .orbit_wars.features_jax import ObsBatch, extract_obs_v9_jax

MAX_PPO_LAUNCH_SLOTS = 16
N_SHIP_OPTIONS = 3


class Transition(NamedTuple):
    done: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: ObsBatch
    action_tgt: jnp.ndarray
    action_frac: jnp.ndarray
    log_prob: jnp.ndarray


class TrainState(NamedTuple):
    model: GraphTransformerV9
    opt_state: optax.OptState


def _normalized_entropy(probs, log_probs, valid_mask):
    """Return entropy normalized by the valid categorical support size."""
    raw_ent = -jnp.sum(probs * log_probs, axis=-1)
    valid_count = jnp.sum(valid_mask.astype(jnp.float32), axis=-1)
    denom = jnp.where(valid_count > 1.0, jnp.log(valid_count), 1.0)
    return jnp.where(valid_count > 1.0, raw_ent / denom, 0.0)


def _weighted_launch_entropy(tgt_probs, tgt_log_probs, tgt_mask, frac_probs, frac_log_probs, frac_mask):
    target_weight = 1.0
    frac_weight = 0.35
    tgt_ent = _normalized_entropy(tgt_probs, tgt_log_probs, tgt_mask)
    frac_ent = _normalized_entropy(frac_probs, frac_log_probs, frac_mask)
    return target_weight * tgt_ent + frac_weight * frac_ent


def _send_logit_bias():
    return 0.0


def forward_and_sample_v9_compact(
    model: GraphTransformerV9,
    obs: ObsBatch,
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
    entropy = jnp.sum(jnp.where(slot_valid, launch_ent, 0.0)) / num_owned

    value = model.value_head(obs.node_features, obs.future_sight, obs.global_features, obs.edge_features)
    action_tgt = jnp.where(slot_valid & (action_send == 1), action_tgt, -1)
    action_frac = jnp.where(slot_valid & (action_send == 1), action_frac, 0)
    return action_tgt, action_frac, step_lp, value.squeeze(), entropy


def compute_log_prob_v9_compact(
    model: GraphTransformerV9,
    obs: ObsBatch,
    action_tgt: jnp.ndarray,
    action_frac: jnp.ndarray,
    temperature: jnp.ndarray = 1.0,
    apm_expected_temperature: jnp.ndarray = 1.0,
) -> tuple:
    """Exact PPO log-prob for taken actions without all-slot pair heads."""
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
    entropy = jnp.sum(jnp.where(slot_valid, launch_ent, 0.0)) / num_owned

    value = model.value_head(obs.node_features, obs.future_sight, obs.global_features, obs.edge_features)
    prior_mu_log, prior_sigma_log = model.launch_prior_head(
        obs.node_features, obs.future_sight, obs.global_features
    )
    return (
        step_lp,
        value.squeeze(),
        entropy,
        p_launch_policy,
        prior_mu_log,
        prior_sigma_log,
    )


def actions_to_env_format(action_tgt: jnp.ndarray, action_frac: jnp.ndarray) -> jnp.ndarray:
    """Convert target and ship-bin actions to flat action indices for jax_orbit_wars_step."""
    flat = action_tgt * N_SHIP_OPTIONS + action_frac
    flat = jnp.where(action_tgt >= 0, flat, -1)
    return flat.astype(jnp.int32)


def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    
    env = OrbitWarsPureJaxEnv(
        episode_steps=config.get("EPISODE_STEPS", 500),
        ship_speed=config.get("SHIP_SPEED", 6.0)
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(frac, 0.0)

    if config.get("ANNEAL_LR", True):
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    def init_fn(rng):
        rng, _rng = jax.random.split(rng)
        
        # Initialize Equinox model
        model = GraphTransformerV9(
            hidden_dim=config.get("D_MODEL", 64),
            n_layers=config.get("NUM_LAYERS", 5),
            heads=config.get("NUM_HEADS", 4),
            n_ship_options=3,
            node_input_dim=21,
            edge_input_dim=14,
            edge_dim=16,
            n_policy_heads=1,
            key=_rng
        )
        
        opt_state = tx.init(eqx.filter(model, eqx.is_array))
        
        train_state = TrainState(
            model=model,
            opt_state=opt_state
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        return (train_state, env_state, obsv, rng)

    def update_fn(runner_state, unused=None):
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION FOR PLAYER 0
                sample_fn = jax.vmap(forward_and_sample_v9_compact, in_axes=(None, 0, 0, None))
                rng, rng_sample = jax.random.split(rng)
                sample_keys = jax.random.split(rng_sample, config["NUM_ENVS"])
                action_tgt, action_frac, step_lp, value, entropy = sample_fn(
                    train_state.model, last_obs, sample_keys, 1.0
                )
                
                # Format to JAX env actions
                actions_p0 = actions_to_env_format(action_tgt, action_frac)
                owned_p0 = last_obs.owned_nodes
                
                # Opponent is no-op
                actions_p1 = jnp.full(actions_p0.shape, -1, dtype=jnp.int32)
                owned_p1 = jnp.full(owned_p0.shape, -1, dtype=jnp.int32)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                step_rngs = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    step_rngs, env_state, actions_p0, owned_p0, actions_p1, owned_p1
                )
                
                transition = Transition(
                    done=done,
                    value=value,
                    reward=reward,
                    obs=last_obs,
                    action_tgt=action_tgt,
                    action_frac=action_frac,
                    log_prob=step_lp,
                )
                
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            
            # Get last value estimate
            sample_fn = jax.vmap(forward_and_sample_v9_compact, in_axes=(None, 0, 0, None))
            rng, rng_sample = jax.random.split(rng)
            sample_keys = jax.random.split(rng_sample, config["NUM_ENVS"])
            _, _, _, last_val, _ = sample_fn(
                train_state.model, last_obs, sample_keys, 1.0
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(model, traj_batch, gae, targets):
                        def loss_single(obs, act_tgt, act_frac, old_lp, adv, target):
                            step_lp, val, ent, _prior_lp, _mu, _sig = compute_log_prob_v9_compact(
                                model, obs, act_tgt, act_frac, 1.0, 1.0
                            )
                            
                            log_ratio = jnp.clip(step_lp - old_lp, -20.0, 20.0)
                            ratio = jnp.exp(log_ratio)
                            surr1 = ratio * adv
                            surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv
                            loss_actor = -jnp.minimum(surr1, surr2)
                            
                            value_loss = 0.5 * jnp.square(val - target)
                            
                            approx_kl = ratio - 1.0 - log_ratio
                            clip_frac = (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).astype(jnp.float32)
                            
                            return loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * ent, (value_loss, loss_actor, ent, approx_kl, clip_frac)
                            
                        losses, aux = jax.vmap(loss_single)(
                            traj_batch.obs,
                            traj_batch.action_tgt,
                            traj_batch.action_frac,
                            traj_batch.log_prob,
                            gae,
                            targets
                        )
                        
                        metrics = (aux[0].mean(), aux[1].mean(), aux[2].mean(), aux[3].mean(), aux[4].mean())
                        return losses.mean(), metrics

                    def _loss_fn_wrapper(model_params, model_static, traj_batch, gae, targets):
                        model = eqx.combine(model_params, model_static)
                        return _loss_fn(model, traj_batch, gae, targets)

                    # GAE Normalization
                    gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    model_params, model_static = eqx.partition(train_state.model, eqx.is_array)
                    
                    grad_fn = jax.value_and_grad(_loss_fn_wrapper, has_aux=True)
                    (total_loss, metrics), grads = grad_fn(model_params, model_static, traj_batch, gae, targets)
                    
                    updates, opt_state = tx.update(grads, train_state.opt_state, model_params)
                    model = eqx.apply_updates(train_state.model, updates)
                    
                    new_train_state = TrainState(model=model, opt_state=opt_state)
                    return new_train_state, (total_loss, metrics)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                
                train_state, minibatches_metrics = jax.lax.scan(_update_minbatch, train_state, minibatches)
                
                # Average metrics over minibatches
                mean_metrics = jax.tree_util.tree_map(lambda x: x.mean(), minibatches_metrics)
                
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, mean_metrics

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]

            # Average metrics over epochs
            mean_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            
            # Combine metrics
            step_metrics = {
                "loss": mean_loss_info[0],
                "value_loss": mean_loss_info[1][0],
                "policy_loss": mean_loss_info[1][1],
                "entropy": mean_loss_info[1][2],
                "approx_kl": mean_loss_info[1][3],
                "clip_frac": mean_loss_info[1][4],
                "explained_variance": jnp.array(0.0), # Statically 0.0 or compute it if needed
                "reward": traj_batch.reward.mean(),
            }

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, step_metrics

    return init_fn, update_fn
