"""PlanetEdgeTransformerJax — GPU-optimized with half-dim edges.

Node hidden_dim=64, Edge edge_dim=32.
Batch matmuls throughout, no vmapped linears.
"""

import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx


class SwiGLUBlock(eqx.Module):
    w_in: Array    # (2H, D)
    b_in: Array    # (2H,)
    w_out: Array   # (D, H)
    b_out: Array   # (D,)

    def __init__(self, dim: int, hidden_dim: int, key):
        k1, k2 = jax.random.split(key)
        self.w_in = jax.random.normal(k1, (hidden_dim * 2, dim)) * (2.0 / dim) ** 0.5
        self.b_in = jnp.zeros(hidden_dim * 2)
        self.w_out = jax.random.normal(k2, (dim, hidden_dim)) * (2.0 / hidden_dim) ** 0.5
        self.b_out = jnp.zeros(dim)

    def __call__(self, x: Array) -> Array:
        h = x @ self.w_in.T + self.b_in
        val, gate = jnp.split(h, 2, axis=-1)
        return (val * jax.nn.silu(gate)) @ self.w_out.T + self.b_out


class NodeEncoder(eqx.Module):
    w_proj: Array    # (H, node_input_dim + future_dim + global_dim)
    b_proj: Array    # (H,)
    swiglu: SwiGLUBlock
    w_out: Array     # (H, H)
    b_out: Array     # (H,)

    def __init__(self, hidden_dim: int, key, node_input_dim: int = 12, future_dim: int = 32, global_dim: int = 8):
        k1, k2, k3 = jax.random.split(key, 3)
        input_dim = int(node_input_dim) + int(future_dim) + int(global_dim)
        self.w_proj = jax.random.normal(k1, (hidden_dim, input_dim)) * (2.0 / input_dim) ** 0.5
        self.b_proj = jnp.zeros(hidden_dim)
        self.swiglu = SwiGLUBlock(hidden_dim, hidden_dim, key=k2)
        self.w_out = jax.random.normal(k3, (hidden_dim, hidden_dim)) * (2.0 / hidden_dim) ** 0.5
        self.b_out = jnp.zeros(hidden_dim)

    def __call__(self, node_features: Array, future_sight: Array, global_features: Array) -> Array:
        gf = jnp.repeat(global_features[None, :], 60, axis=0)
        h = jnp.concatenate([node_features, future_sight, gf], axis=-1)
        h = _ln(h)
        h = jax.nn.silu(h @ self.w_proj.T + self.b_proj)
        h = h + self.swiglu(h)
        h = h @ self.w_out.T + self.b_out
        return _ln(h)


class EdgeEncoder(eqx.Module):
    """edge_input_dim -> edge_dim (half of hidden_dim by default)."""
    w1: Array    # (E, edge_input_dim)
    b1: Array    # (E,)
    w2: Array    # (E, E)
    b2: Array    # (E,)
    edge_input_dim: int = eqx.field(static=True)

    def __init__(self, edge_dim: int, key, edge_input_dim: int = 9):
        k1, k2 = jax.random.split(key)
        self.edge_input_dim = int(edge_input_dim)
        self.w1 = jax.random.normal(k1, (edge_dim, self.edge_input_dim)) * (2.0 / self.edge_input_dim) ** 0.5
        self.b1 = jnp.zeros(edge_dim)
        self.w2 = jax.random.normal(k2, (edge_dim, edge_dim)) * (2.0 / edge_dim) ** 0.5
        self.b2 = jnp.zeros(edge_dim)

    def __call__(self, edge_features: Array) -> Array:
        flat = edge_features.reshape(-1, self.edge_input_dim)
        h = jax.nn.silu(_ln(flat) @ self.w1.T + self.b1)
        h = _ln(h) @ self.w2.T + self.b2
        return h.reshape(60, 60, -1)                # (60, 60, E)


class EdgeBiasedTransformerLayer(eqx.Module):
    hidden_dim: int = eqx.field(static=True)
    edge_dim: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    has_edge_update: bool = eqx.field(static=True)

    # QKV (node dim)
    w_q: Array; b_q: Array
    w_k: Array; b_k: Array
    w_v: Array; b_v: Array
    w_out: Array; b_out: Array
    # Edge bias (edge_dim -> heads)
    w_ebias: Array    # (heads, E)
    # Node FFN
    w_ffn1: Array; b_ffn1: Array
    w_ffn2: Array; b_ffn2: Array
    # Edge update: concat[edge_h(60,60,E), src(60,60,H), dst(60,60,H)] -> E
    w_edge1: Array; b_edge1: Array
    w_edge2: Array; b_edge2: Array
    # LayerNorm weights
    ln_w: Array; ln_b: Array
    ffn_ln_w: Array; ffn_ln_b: Array
    edge_ln_w: Array; edge_ln_b: Array

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int, ffn_mult: int, update_edges: bool, key):
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim ** -0.5
        self.has_edge_update = update_edges

        keys = jax.random.split(key, 14)
        ki = iter(keys)
        H, E = hidden_dim, edge_dim

        self.ln_w = jnp.ones(H); self.ln_b = jnp.zeros(H)
        self.w_q = _wi(next(ki), (H, H)); self.b_q = jnp.zeros(H)
        self.w_k = _wi(next(ki), (H, H)); self.b_k = jnp.zeros(H)
        self.w_v = _wi(next(ki), (H, H)); self.b_v = jnp.zeros(H)
        self.w_ebias = _wi(next(ki), (heads, E))
        self.w_out = _wi(next(ki), (H, H)); self.b_out = jnp.zeros(H)

        self.ffn_ln_w = jnp.ones(H); self.ffn_ln_b = jnp.zeros(H)
        ffn_dim = H * ffn_mult
        self.w_ffn1 = _wi(next(ki), (ffn_dim, H)); self.b_ffn1 = jnp.zeros(ffn_dim)
        self.w_ffn2 = _wi(next(ki), (H, ffn_dim)); self.b_ffn2 = jnp.zeros(H)

        edge_update_in = E + 2 * H  # concat[edge_h, src, dst]
        self.edge_ln_w = jnp.ones(edge_update_in); self.edge_ln_b = jnp.zeros(edge_update_in)
        self.w_edge1 = _wi(next(ki), (ffn_dim, edge_update_in)); self.b_edge1 = jnp.zeros(ffn_dim)
        self.w_edge2 = _wi(next(ki), (E, ffn_dim)); self.b_edge2 = jnp.zeros(E)

    def __call__(self, node_h: Array, edge_h: Array) -> tuple[Array, Array]:
        N, H, E = 60, self.hidden_dim, self.edge_dim

        q_norm = _ln(node_h, self.ln_w, self.ln_b)
        q = (q_norm @ self.w_q.T + self.b_q).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)
        k = (q_norm @ self.w_k.T + self.b_k).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)
        v = (q_norm @ self.w_v.T + self.b_v).reshape(N, self.heads, self.head_dim).transpose(1, 0, 2)

        scores = jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale
        bias = (edge_h.reshape(-1, E) @ self.w_ebias.T).reshape(N, N, self.heads).transpose(2, 0, 1)
        attn = jax.nn.softmax(scores + bias, axis=-1)
        context = jnp.matmul(attn, v).transpose(1, 0, 2).reshape(N, H)

        node_h = node_h + (context @ self.w_out.T + self.b_out)
        h_norm = _ln(node_h, self.ffn_ln_w, self.ffn_ln_b)
        node_h = node_h + (jax.nn.silu(h_norm @ self.w_ffn1.T + self.b_ffn1) @ self.w_ffn2.T + self.b_ffn2)

        if self.has_edge_update:
            src = jnp.repeat(node_h[:, None, :], N, axis=1)    # (60, 60, H)
            dst = jnp.repeat(node_h[None, :, :], N, axis=0)    # (60, 60, H)
            cat = jnp.concatenate([edge_h, src, dst], axis=-1)  # (60, 60, E+2H)
            flat = cat.reshape(-1, E + 2 * H)
            flat = _ln(flat, self.edge_ln_w, self.edge_ln_b)
            edge_h = edge_h + (jax.nn.silu(flat @ self.w_edge1.T + self.b_edge1) @ self.w_edge2.T + self.b_edge2).reshape(N, N, E)

        return node_h, edge_h


class AttentionPool(eqx.Module):
    w: Array  # (1, H)
    def __init__(self, H, key):
        self.w = _wi(key, (1, H))
    def __call__(self, node_h):
        s = (node_h @ self.w.T).squeeze(-1)
        w = jax.nn.softmax(s, axis=0)
        return (node_h * w[:, None]).sum(axis=0)


class ValueHead(eqx.Module):
    pool: AttentionPool
    w1: Array; b1: Array
    w_gate: Array; b_gate: Array
    w3: Array; b3: Array

    def __init__(self, H, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.pool = AttentionPool(H, key=k1)
        d = H + 8
        self.w1 = _wi(k1, (H, d)); self.b1 = jnp.zeros(H)
        self.w_gate = _wi(k2, (2 * H, H)); self.b_gate = jnp.zeros(2 * H)
        self.w3 = _wi(k3, (1, H)); self.b3 = jnp.zeros(1)

    def __call__(self, node_h, gf):
        p = self.pool(node_h)
        h = jnp.concatenate([p, gf])
        h = jax.nn.silu(_ln_scalar(h) @ self.w1.T + self.b1)
        gh = h @ self.w_gate.T + self.b_gate
        val, gate = jnp.split(gh, 2, axis=-1)
        return jnp.tanh((val * jax.nn.sigmoid(gate) @ self.w3.T + self.b3).squeeze(-1))


class TinyValueNet(eqx.Module):
    """Small critic-only encoder, fully separate from the policy trunk."""
    w_proj: Array
    b_proj: Array
    layers: list
    pool: AttentionPool
    w1: Array
    b1: Array
    w2: Array
    b2: Array

    def __init__(self, hidden_dim: int, n_layers: int, heads: int, ffn_mult: int, key, node_input_dim: int = 12):
        keys = jax.random.split(key, 4 + max(n_layers, 0))
        input_dim = int(node_input_dim) + 32 + 8
        self.w_proj = _wi(keys[0], (hidden_dim, input_dim))
        self.b_proj = jnp.zeros(hidden_dim)
        self.layers = [
            EdgeBiasedTransformerLayer(
                hidden_dim,
                max(1, hidden_dim // 2),
                heads,
                ffn_mult,
                update_edges=False,
                key=lk,
            )
            for lk in keys[4:]
        ]
        self.pool = AttentionPool(hidden_dim, key=keys[1])
        self.w1 = _wi(keys[2], (hidden_dim, hidden_dim + 8))
        self.b1 = jnp.zeros(hidden_dim)
        self.w2 = _wi(keys[3], (1, hidden_dim))
        self.b2 = jnp.zeros(1)

    def __call__(self, node_features, future_sight, global_features):
        gf = jnp.repeat(global_features[None, :], 60, axis=0)
        h = jnp.concatenate([node_features, future_sight, gf], axis=-1)
        h = jax.nn.silu(_ln(h) @ self.w_proj.T + self.b_proj)
        dummy_edge = jnp.zeros((60, 60, max(1, h.shape[-1] // 2)), dtype=h.dtype)
        for layer in self.layers:
            h, dummy_edge = layer(h, dummy_edge)
        p = self.pool(_ln(h))
        h = jnp.concatenate([p, global_features])
        h = jax.nn.silu(_ln_scalar(h) @ self.w1.T + self.b1)
        return jnp.tanh((h @ self.w2.T + self.b2).squeeze(-1))


class PolicyEdgeHead(eqx.Module):
    w_send1: Array; b_send1: Array
    w_send2: Array; b_send2: Array
    w_tgt1: Array; b_tgt1: Array
    w_tgt2: Array; b_tgt2: Array
    w_frac1: Array; b_frac1: Array
    w_frac2: Array; b_frac2: Array

    n_ship_options: int

    def __init__(self, H, E, n_ship_options: int = 3, key=None):
        k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        d = 2 * H + E + 8  # src + dst + edge + global
        send_d = H + 8
        self.n_ship_options = int(n_ship_options)
        self.w_send1 = _wi(k0, (H, send_d)); self.b_send1 = jnp.zeros(H)
        self.w_send2 = _wi(k1, (1, H)); self.b_send2 = jnp.zeros(1)
        self.w_tgt1 = _wi(k2, (H, d)); self.b_tgt1 = jnp.zeros(H)
        self.w_tgt2 = _wi(k3, (1, H)); self.b_tgt2 = jnp.zeros(1)
        self.w_frac1 = _wi(k4, (H, d)); self.b_frac1 = jnp.zeros(H)
        self.w_frac2 = _wi(k5, (self.n_ship_options, H)); self.b_frac2 = jnp.zeros(self.n_ship_options)

    def __call__(self, node_h, edge_h, owned, gf):
        N, H = 60, node_h.shape[-1]
        E = edge_h.shape[-1]
        src_safe = jnp.where(owned >= 0, owned, 0)
        src_h = node_h[src_safe]                     # (M, H)
        edge_s = edge_h[src_safe]                     # (M, N, E)
        M = owned.shape[0]
        send_cat = jnp.concatenate([
            src_h,
            jnp.broadcast_to(gf[None, :], (M, 8)),
        ], axis=-1)
        send = (jax.nn.silu(send_cat @ self.w_send1.T + self.b_send1) @ self.w_send2.T + self.b_send2).squeeze(-1)

        cat = jnp.concatenate([
            jnp.broadcast_to(src_h[:, None, :], (M, N, H)),
            jnp.broadcast_to(node_h[None, :, :], (M, N, H)),
            edge_s,
            jnp.broadcast_to(gf[None, None, :], (M, N, 8)),
        ], axis=-1)  # (M, N, 2H+E+8)

        flat = cat.reshape(-1, H + H + E + 8)

        tgt = (jax.nn.silu(flat @ self.w_tgt1.T + self.b_tgt1) @ self.w_tgt2.T + self.b_tgt2).reshape(M, N)
        frac = (jax.nn.silu(flat @ self.w_frac1.T + self.b_frac1) @ self.w_frac2.T + self.b_frac2).reshape(M, N, self.n_ship_options)
        return send, tgt, frac


class LaunchCountPriorNet(eqx.Module):
    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array
    w4: Array
    b4: Array
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int = 32, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.hidden_dim = int(hidden_dim)
        # Explicit global/prior scalars, matching ONLINE_PRIOR_FEATURES in train_jax.py.
        in_dim = 16
        self.w1 = _wi(k1, (self.hidden_dim, in_dim))
        self.b1 = jnp.zeros(self.hidden_dim)
        self.w2 = _wi(k2, (self.hidden_dim, self.hidden_dim))
        self.b2 = jnp.zeros(self.hidden_dim)
        self.w3 = _wi(k3, (self.hidden_dim, self.hidden_dim))
        self.b3 = jnp.zeros(self.hidden_dim)
        self.w4 = _wi(k4, (2, self.hidden_dim))
        self.b4 = jnp.zeros(2)

    def __call__(self, node_features, future_sight, global_features):
        del future_sight
        max_p = 60.0
        max_ships = 400.0
        max_prod = 5.0
        denom = max_p * max_ships
        our_mask = node_features[:, 0] > 0.5
        opp_mask = node_features[:, 1] > 0.5
        planet_ships = node_features[:, 6] * max_ships
        production = node_features[:, 7] * max_prod

        tick_norm = global_features[0]
        our_n_planets = global_features[1] * max_p
        opp_n_planets = global_features[2] * max_p
        neutral_n_planets = global_features[3] * max_p
        our_ships_planets = global_features[4] * denom
        opp_ships_planets = global_features[5] * denom
        our_ships_fleets = global_features[6] * denom
        opp_ships_fleets = global_features[7] * denom
        our_production = jnp.sum(jnp.where(our_mask, production, 0.0))
        opp_production = jnp.sum(jnp.where(opp_mask, production, 0.0))
        our_total_ships = our_ships_planets + our_ships_fleets
        opp_total_ships = opp_ships_planets + opp_ships_fleets
        our_ship_entropy = _ship_entropy(planet_ships, our_mask)
        opp_ship_entropy = _ship_entropy(planet_ships, opp_mask)
        ship_advantage_log = jnp.log1p(our_total_ships) - jnp.log1p(opp_total_ships)
        planet_advantage = our_n_planets - opp_n_planets
        production_advantage = our_production - opp_production
        fleet_pressure = jnp.log1p(opp_ships_fleets) - jnp.log1p(our_ships_fleets)

        x = jnp.stack([
            tick_norm,
            our_n_planets,
            opp_n_planets,
            neutral_n_planets,
            our_production,
            opp_production,
            our_ships_planets,
            opp_ships_planets,
            our_ships_fleets,
            opp_ships_fleets,
            our_ship_entropy,
            opp_ship_entropy,
            ship_advantage_log,
            planet_advantage,
            production_advantage,
            fleet_pressure,
        ])
        h = jax.nn.silu(_ln_scalar(x) @ self.w1.T + self.b1)
        h = jax.nn.silu(h @ self.w2.T + self.b2)
        h = jax.nn.silu(h @ self.w3.T + self.b3)
        raw = h @ self.w4.T + self.b4
        mu_log = jax.nn.sigmoid(raw[0]) * jnp.log1p(60.0)
        sigma_log = jax.nn.softplus(raw[1]) + 0.20
        return mu_log, sigma_log


class PlanetEdgeTransformerJax(eqx.Module):
    node_encoder: NodeEncoder
    edge_encoder: EdgeEncoder
    layers: list
    policy_head: PolicyEdgeHead
    value_head: TinyValueNet
    launch_prior_head: LaunchCountPriorNet
    node_ln_w: Array; node_ln_b: Array
    edge_ln_w: Array; edge_ln_b: Array

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        heads: int = 4,
        ffn_mult: int = 2,
        value_hidden_dim: int = 32,
        value_layers: int = 3,
        prior_hidden_dim: int = 32,
        n_ship_options: int = 3,
        edge_input_dim: int = 9,
        key = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        edge_dim = hidden_dim // 2
        H, E = hidden_dim, edge_dim

        self.node_encoder = NodeEncoder(H, key=k1)
        self.edge_encoder = EdgeEncoder(E, key=k2, edge_input_dim=edge_input_dim)
        self.layers = [
            EdgeBiasedTransformerLayer(H, E, heads, ffn_mult, update_edges=(i == n_layers - 1), key=lk)
            for i, lk in enumerate(jax.random.split(k3, n_layers))
        ]
        self.node_ln_w = jnp.ones(H); self.node_ln_b = jnp.zeros(H)
        self.edge_ln_w = jnp.ones(E); self.edge_ln_b = jnp.zeros(E)
        self.policy_head = PolicyEdgeHead(H, E, n_ship_options=n_ship_options, key=k4)
        self.value_head = TinyValueNet(
            value_hidden_dim,
            value_layers,
            max(1, min(heads, value_hidden_dim)),
            ffn_mult,
            key=k5,
        )
        self.launch_prior_head = LaunchCountPriorNet(prior_hidden_dim, key=k6)

    def __call__(self, node_features, edge_features, future_sight, global_features, owned_nodes):
        node_h = self.node_encoder(node_features, future_sight, global_features)  # (60, H)
        edge_h = self.edge_encoder(edge_features)                                  # (60, 60, E)
        # Stop gradient on edge_h: edges provide attention bias in the forward pass,
        # but we don't backprop through them. This prevents OOM from storing
        # (batch, 3600, 160) intermediates for the edge update block during backprop.
        edge_h = jax.lax.stop_gradient(edge_h)
        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h)

        node_h = _ln(node_h, self.node_ln_w, self.node_ln_b)
        eflat = edge_h.reshape(-1, edge_h.shape[-1])
        edge_h = _ln(eflat, self.edge_ln_w, self.edge_ln_b).reshape(edge_h.shape)

        send, tgt, frac = self.policy_head(node_h, edge_h, owned_nodes, global_features)
        val = self.value_head(node_features, future_sight, global_features)
        prior_mu_log, prior_sigma_log = self.launch_prior_head(node_features, future_sight, global_features)
        return send, tgt, frac, prior_mu_log, prior_sigma_log, val


# ── helpers ──

def _wi(key, shape):
    return jax.random.normal(key, shape) * (2.0 / shape[1]) ** 0.5

def _ln(x, w=None, b=None):
    m = x.mean(axis=-1, keepdims=True)
    var = jnp.mean((x - m) * (x - m), axis=-1, keepdims=True)
    s = jnp.sqrt(var + 1e-5)
    out = (x - m) / s
    if w is not None:
        out = out * w + b
    return out

def _ln_scalar(x):
    m = x.mean()
    var = jnp.mean((x - m) * (x - m))
    return (x - m) / jnp.sqrt(var + 1e-5)

def _ship_entropy(ships, mask):
    mask_f = mask.astype(jnp.float32)
    total = jnp.sum(ships * mask_f)
    count = jnp.sum(mask_f)
    slog = jnp.sum(jnp.where(mask, ships * jnp.log(jnp.maximum(ships, 1.0)), 0.0))
    raw = jnp.log(jnp.maximum(total, 1.0)) - slog / jnp.maximum(total, 1.0)
    denom = jnp.log(jnp.maximum(count, 2.0))
    return jnp.where((total > 0.0) & (count > 1.0), raw / denom, 0.0)
