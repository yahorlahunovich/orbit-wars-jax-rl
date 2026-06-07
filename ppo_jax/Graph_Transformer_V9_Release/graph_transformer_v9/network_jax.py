"""GraphTransformerV9 network.

V9 removes the all-pairs edge-aware transformer blocks from V8. The policy
trunk is a normal node transformer that remains fully trainable during PPO.
Edge facts are encoded by a tiny MLP and consumed directly by the send/target/
bucket heads. The value network is fully separate and receives a lightweight
edge summary.
"""

from __future__ import annotations

import operator

import equinox as eqx
import jax
import jax.numpy as jnp

from models.planet_transformer_jax.network_jax import (
    AttentionPool,
    EdgeEncoder,
    LaunchCountPriorNet,
    NodeEncoder,
    _ln,
    _ln_scalar,
    _wi,
)


class NodeTransformerLayer(eqx.Module):
    hidden_dim: int
    heads: int
    head_dim: int
    scale: float
    w_q: jax.Array
    b_q: jax.Array
    w_k: jax.Array
    b_k: jax.Array
    w_v: jax.Array
    b_v: jax.Array
    w_out: jax.Array
    b_out: jax.Array
    w_ffn1: jax.Array
    b_ffn1: jax.Array
    w_ffn2: jax.Array
    b_ffn2: jax.Array
    ln_w: jax.Array
    ln_b: jax.Array
    ffn_ln_w: jax.Array
    ffn_ln_b: jax.Array

    def __init__(self, hidden_dim: int, heads: int, ffn_mult: int, key):
        keys = jax.random.split(key, 8)
        H = int(hidden_dim)
        self.hidden_dim = H
        self.heads = int(heads)
        self.head_dim = H // int(heads)
        self.scale = self.head_dim ** -0.5
        self.ln_w = jnp.ones(H)
        self.ln_b = jnp.zeros(H)
        self.w_q = _wi(keys[0], (H, H))
        self.b_q = jnp.zeros(H)
        self.w_k = _wi(keys[1], (H, H))
        self.b_k = jnp.zeros(H)
        self.w_v = _wi(keys[2], (H, H))
        self.b_v = jnp.zeros(H)
        self.w_out = _wi(keys[3], (H, H))
        self.b_out = jnp.zeros(H)
        self.ffn_ln_w = jnp.ones(H)
        self.ffn_ln_b = jnp.zeros(H)
        ffn_dim = H * int(ffn_mult)
        self.w_ffn1 = _wi(keys[4], (ffn_dim, H))
        self.b_ffn1 = jnp.zeros(ffn_dim)
        self.w_ffn2 = _wi(keys[5], (H, ffn_dim))
        self.b_ffn2 = jnp.zeros(H)

    def __call__(self, node_h):
        N, H = node_h.shape
        q_norm = _ln(node_h, self.ln_w, self.ln_b)
        q = (q_norm @ self.w_q.T + self.b_q).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)
        k = (q_norm @ self.w_k.T + self.b_k).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)
        v = (q_norm @ self.w_v.T + self.b_v).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.matmul(attn, v).transpose(1, 0, 2).reshape(N, H)
        node_h = node_h + (ctx @ self.w_out.T + self.b_out)
        h_norm = _ln(node_h, self.ffn_ln_w, self.ffn_ln_b)
        node_h = node_h + (jax.nn.silu(h_norm @ self.w_ffn1.T + self.b_ffn1) @ self.w_ffn2.T + self.b_ffn2)
        return node_h


class EdgeAwareValueNet(eqx.Module):
    value_uses_edge: bool = eqx.field(static=True, default=True)
    w_proj: jax.Array
    b_proj: jax.Array
    layers: list
    edge_encoder: EdgeEncoder
    pool: AttentionPool
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        heads: int,
        ffn_mult: int,
        edge_dim: int,
        edge_input_dim: int,
        key,
        node_input_dim: int = 21,
    ):
        keys = jax.random.split(key, 5 + max(n_layers, 0))
        input_dim = int(node_input_dim) + 32 + 8
        self.w_proj = _wi(keys[0], (hidden_dim, input_dim))
        self.b_proj = jnp.zeros(hidden_dim)
        self.layers = [
            NodeTransformerLayer(hidden_dim, heads, ffn_mult, key=lk)
            for lk in keys[5:]
        ]
        self.edge_encoder = EdgeEncoder(edge_dim, key=keys[1], edge_input_dim=edge_input_dim)
        self.pool = AttentionPool(hidden_dim, key=keys[2])
        self.w1 = _wi(keys[3], (hidden_dim, hidden_dim + edge_dim + 8))
        self.b1 = jnp.zeros(hidden_dim)
        self.w2 = _wi(keys[4], (1, hidden_dim))
        self.b2 = jnp.zeros(1)

    def __call__(self, node_features, future_sight, global_features, edge_features=None):
        gf = jnp.repeat(global_features[None, :], 60, axis=0)
        h = jnp.concatenate([node_features, future_sight, gf], axis=-1)
        h = jax.nn.silu(_ln(h) @ self.w_proj.T + self.b_proj)
        for layer in self.layers:
            h = layer(h)
        node_summary = self.pool(_ln(h))
        if edge_features is None:
            edge_summary = jnp.zeros(self.edge_encoder.w2.shape[0], dtype=h.dtype)
        else:
            edge_h = self.edge_encoder(edge_features)
            edge_summary = jnp.mean(edge_h, axis=(0, 1))
        z = jnp.concatenate([node_summary, edge_summary, global_features])
        z = jax.nn.silu(_ln_scalar(z) @ self.w1.T + self.b1)
        return jnp.tanh((z @ self.w2.T + self.b2).squeeze(-1))


class V9PolicyHead(eqx.Module):
    w_send1: jax.Array
    b_send1: jax.Array
    w_send2: jax.Array
    b_send2: jax.Array
    w_tgt1: jax.Array
    b_tgt1: jax.Array
    w_tgt2: jax.Array
    b_tgt2: jax.Array
    w_bucket1: jax.Array
    b_bucket1: jax.Array
    w_bucket2: jax.Array
    b_bucket2: jax.Array
    w_frac1: jax.Array
    b_frac1: jax.Array
    w_frac2: jax.Array
    b_frac2: jax.Array
    n_ship_options: int

    def __init__(self, hidden_dim: int, edge_dim: int, n_ship_options: int = 3, key=None):
        k0, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)
        H, E = int(hidden_dim), int(edge_dim)
        self.n_ship_options = int(n_ship_options)
        send_d = H + 8
        pair_d = 2 * H + E + 8
        bucket_raw_d = 4
        bucket_d = pair_d + bucket_raw_d
        self.w_send1 = _wi(k0, (H, send_d))
        self.b_send1 = jnp.zeros(H)
        self.w_send2 = _wi(k1, (1, H))
        self.b_send2 = jnp.zeros(1)
        self.w_tgt1 = _wi(k2, (H, pair_d))
        self.b_tgt1 = jnp.zeros(H)
        self.w_tgt2 = _wi(k3, (1, H))
        self.b_tgt2 = jnp.zeros(1)
        self.w_bucket1 = _wi(k4, (H, bucket_d))
        self.b_bucket1 = jnp.zeros(H)
        self.w_bucket2 = _wi(k5, (1, H))
        self.b_bucket2 = jnp.zeros(1)
        self.w_frac1 = _wi(k6, (H, bucket_d))
        self.b_frac1 = jnp.zeros(H)
        self.w_frac2 = _wi(k7, (1, H))
        self.b_frac2 = jnp.zeros(1)

    def _bucket_raw(self, raw_edge_s):
        roi = raw_edge_s[:, :, 6:7]
        ratio = raw_edge_s[:, :, 7:8]
        turns = raw_edge_s[:, :, 8:11]
        clear = raw_edge_s[:, :, 11:14]
        per_bucket = [
            jnp.concatenate([roi, ratio, turns[:, :, i:i + 1], clear[:, :, i:i + 1]], axis=-1)
            for i in range(self.n_ship_options)
        ]
        return jnp.stack(per_bucket, axis=2).reshape(
            raw_edge_s.shape[0], raw_edge_s.shape[1], self.n_ship_options, 4
        )

    def __call__(self, node_h, edge_h, raw_edge, owned, gf):
        N, H = node_h.shape
        E = edge_h.shape[-1]
        src_safe = jnp.where(owned >= 0, owned, 0)
        src_h = node_h[src_safe]
        edge_s = edge_h[src_safe]
        raw_edge_s = raw_edge[src_safe]
        M = owned.shape[0]

        send_cat = jnp.concatenate([src_h, jnp.broadcast_to(gf[None, :], (M, 8))], axis=-1)
        send = (jax.nn.silu(send_cat @ self.w_send1.T + self.b_send1) @ self.w_send2.T + self.b_send2).squeeze(-1)

        pair_cat = jnp.concatenate(
            [
                jnp.broadcast_to(src_h[:, None, :], (M, N, H)),
                jnp.broadcast_to(node_h[None, :, :], (M, N, H)),
                edge_s,
                jnp.broadcast_to(gf[None, None, :], (M, N, 8)),
            ],
            axis=-1,
        )
        flat_pair = pair_cat.reshape(-1, 2 * H + E + 8)
        target_base = (
            jax.nn.silu(flat_pair @ self.w_tgt1.T + self.b_tgt1) @ self.w_tgt2.T + self.b_tgt2
        ).reshape(M, N)

        bucket_raw = self._bucket_raw(raw_edge_s)
        bucket_cat = jnp.concatenate(
            [
                jnp.broadcast_to(pair_cat[:, :, None, :], (M, N, self.n_ship_options, 2 * H + E + 8)),
                bucket_raw,
            ],
            axis=-1,
        )
        flat_bucket = bucket_cat.reshape(-1, 2 * H + E + 8 + 4)
        bucket_utility = (
            jax.nn.silu(flat_bucket @ self.w_bucket1.T + self.b_bucket1) @ self.w_bucket2.T + self.b_bucket2
        ).reshape(M, N, self.n_ship_options)
        target = target_base + jnp.max(bucket_utility, axis=-1)
        frac = (
            jax.nn.silu(flat_bucket @ self.w_frac1.T + self.b_frac1) @ self.w_frac2.T + self.b_frac2
        ).reshape(M, N, self.n_ship_options)
        return send, target, frac


class GraphTransformerV9(eqx.Module):
    value_uses_edge: bool = eqx.field(static=True, default=True)
    node_encoder: NodeEncoder
    edge_encoder: EdgeEncoder
    layers: list
    policy_heads: tuple
    value_head: EdgeAwareValueNet
    launch_prior_head: LaunchCountPriorNet
    node_ln_w: jax.Array
    node_ln_b: jax.Array
    edge_ln_w: jax.Array
    edge_ln_b: jax.Array
    n_policy_heads: int
    edge_dim: int

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 5,
        heads: int = 4,
        ffn_mult: int = 2,
        value_hidden_dim: int = 32,
        value_layers: int = 4,
        prior_hidden_dim: int = 32,
        n_ship_options: int = 3,
        node_input_dim: int = 21,
        edge_input_dim: int = 14,
        edge_dim: int = 16,
        n_policy_heads: int = 10,
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.edge_dim = int(edge_dim)
        self.node_encoder = NodeEncoder(hidden_dim, key=k1, node_input_dim=node_input_dim)
        self.edge_encoder = EdgeEncoder(edge_dim, key=k2, edge_input_dim=edge_input_dim)
        self.layers = [
            NodeTransformerLayer(hidden_dim, heads, ffn_mult, key=lk)
            for lk in jax.random.split(k3, n_layers)
        ]
        self.node_ln_w = jnp.ones(hidden_dim)
        self.node_ln_b = jnp.zeros(hidden_dim)
        self.edge_ln_w = jnp.ones(edge_dim)
        self.edge_ln_b = jnp.zeros(edge_dim)
        self.n_policy_heads = int(n_policy_heads)
        self.policy_heads = tuple(
            V9PolicyHead(hidden_dim, edge_dim, n_ship_options=n_ship_options, key=hk)
            for hk in jax.random.split(k4, self.n_policy_heads)
        )
        self.value_head = EdgeAwareValueNet(
            value_hidden_dim,
            value_layers,
            max(1, min(heads, value_hidden_dim)),
            ffn_mult,
            edge_dim,
            edge_input_dim,
            key=k5,
            node_input_dim=node_input_dim,
        )
        self.launch_prior_head = LaunchCountPriorNet(prior_hidden_dim, key=k6)

    def encode(self, node_features, edge_features, future_sight, global_features):
        node_h = self.node_encoder(node_features, future_sight, global_features)
        for layer in self.layers:
            node_h = layer(node_h)
        node_h = _ln(node_h, self.node_ln_w, self.node_ln_b)
        edge_h = self.edge_encoder(edge_features)
        eflat = edge_h.reshape(-1, edge_h.shape[-1])
        edge_h = _ln(eflat, self.edge_ln_w, self.edge_ln_b).reshape(edge_h.shape)
        return node_h, edge_h

    def __call__(
        self,
        node_features,
        edge_features,
        future_sight,
        global_features,
        owned_nodes,
        player_head_idx=0,
    ):
        node_h, edge_h = self.encode(node_features, edge_features, future_sight, global_features)
        if isinstance(player_head_idx, int):
            head_idx = max(0, min(operator.index(player_head_idx), self.n_policy_heads - 1))
            send, tgt, frac = self.policy_heads[head_idx](node_h, edge_h, edge_features, owned_nodes, global_features)
        else:
            idx = jnp.clip(jnp.asarray(player_head_idx, dtype=jnp.int32), 0, self.n_policy_heads - 1)

            def branch(head):
                return lambda _: head(node_h, edge_h, edge_features, owned_nodes, global_features)

            send, tgt, frac = jax.lax.switch(
                idx,
                tuple(branch(head) for head in self.policy_heads),
                operand=None,
            )
        val = self.value_head(node_features, future_sight, global_features, edge_features)
        prior_mu_log, prior_sigma_log = self.launch_prior_head(
            node_features, future_sight, global_features
        )
        return send, tgt, frac, prior_mu_log, prior_sigma_log, val
