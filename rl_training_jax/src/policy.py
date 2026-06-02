"""Transformer policy for Orbit Wars (JAX/Flax).

Consumes the per-entity feature dict produced by
`orbit_wars.features_jax.encode_observation` and emits:

- target logits over MAX_PLANETS target slots (which planet to attack/reinforce)
- ship-bucket logits over (MAX_PLANETS, BUCKET_COUNT) conditional on target
- a single scalar value (for the critic)

Design (Expert-optimized):

    tokens = [global_token, planet_tokens, fleet_tokens]
    tokens = TransformerEncoder(d_model, n_heads, n_layers)(tokens, mask)
    target_logits = dot_product(planet_h, planet_h) + noop_bias
    bucket_logits = MLP(planet_h_src, planet_h_tgt)
    value         = MLP(global_h)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct


@struct.dataclass
class PolicyOutput:
    target_logits: jnp.ndarray   # (B, MAX_PLANETS, MAX_PLANETS)
    bucket_logits: jnp.ndarray   # (B, MAX_PLANETS, MAX_PLANETS, BUCKET_COUNT)
    value: jnp.ndarray           # (B,)


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_mult: int = 4

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, kv_padding_mask: jnp.ndarray) -> jnp.ndarray:
        # tokens: (B, T, d)
        # kv_padding_mask: (B, T) bool — True for *valid* tokens.
        b, t, _ = tokens.shape

        mask = kv_padding_mask[:, None, None, :]                      # (B, 1, 1, T)
        mask = jnp.broadcast_to(mask, (b, self.num_heads, t, t))

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
        )
        x_norm = nn.LayerNorm()(tokens)
        x_attn = attn(x_norm, x_norm, mask=mask)
        tokens = tokens + x_attn

        y_norm = nn.LayerNorm()(tokens)
        y = nn.Dense(self.d_model * self.ff_mult)(y_norm)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        tokens = tokens + y

        tokens = tokens * kv_padding_mask[:, :, None].astype(tokens.dtype)
        return tokens


class PlanetPolicy(nn.Module):
    """Transformer over (global, planets, fleets) producing planet-action heads."""

    planet_count: int
    fleet_count: int
    bucket_count: int = 8
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ff_mult: int = 4
    noop_bias_init: float = 2.0

    def setup(self) -> None:
        self.global_in = nn.Dense(self.d_model)
        self.planet_in = nn.Dense(self.d_model)
        self.fleet_in = nn.Dense(self.d_model)
        
        # Token-type embeddings: 0=Global, 1=Planet, 2=Fleet
        self.type_emb = nn.Embed(3, self.d_model)
        
        self.blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.ff_mult)
            for _ in range(self.num_layers)
        ]
        
        self.target_proj_q = nn.Dense(self.d_model)
        self.target_proj_k = nn.Dense(self.d_model)
        
        # Bucket head takes concatenated source and target planet representations
        self.bucket_head = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(self.bucket_count)
        ])
        
        self.value_head = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(1)
        ])
        
        self.noop_bias = self.param(
            "noop_bias", nn.initializers.constant(self.noop_bias_init), ()
        )

    def __call__(
        self,
        planet_features: jnp.ndarray,    # (B, P, F_p)
        planet_mask: jnp.ndarray,        # (B, P) bool
        fleet_features: jnp.ndarray,     # (B, F, F_f)
        fleet_mask: jnp.ndarray,         # (B, F) bool
        global_features: jnp.ndarray,    # (B, F_g)
    ) -> PolicyOutput:
        b, p, _ = planet_features.shape
        f = fleet_features.shape[1]

        # 1. Project to d_model
        g_tok = self.global_in(global_features)[:, None, :]  # (B, 1, d)
        p_tok = self.planet_in(planet_features)              # (B, P, d)
        f_tok = self.fleet_in(fleet_features)                # (B, F, d)
        
        # 2. Add Type Embeddings
        type_idx = jnp.concatenate([
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.ones((p,), dtype=jnp.int32),
            jnp.full((f,), 2, dtype=jnp.int32),
        ])
        t_embs = self.type_emb(type_idx)                     # (1+P+F, d)
        
        tokens = jnp.concatenate([g_tok, p_tok, f_tok], axis=1) # (B, 1+P+F, d)
        tokens = tokens + t_embs[None, :, :]
        
        full_mask = jnp.concatenate([
            jnp.ones((b, 1), dtype=jnp.bool_),
            planet_mask,
            fleet_mask
        ], axis=1)                                           # (B, 1+P+F)

        # 3. Transformer
        for block in self.blocks:
            tokens = block(tokens, full_mask)

        # 4. Extract Planet representations
        g_h = tokens[:, 0, :]                                # (B, d)
        p_h = tokens[:, 1 : 1 + p, :]                        # (B, P, d)

        # 5. Target Head (einsum bilinear)
        q = self.target_proj_q(p_h)                          # (B, P, d)
        k = self.target_proj_k(p_h)                          # (B, P, d)
        scale = jnp.float32(1.0 / jnp.sqrt(self.d_model))
        target_logits = jnp.einsum("bsd,btd->bst", q, k) * scale     # (B, P, P)

        # Apply NOOP bias to diagonal
        diag = jnp.eye(p, dtype=target_logits.dtype)
        target_logits = target_logits + diag[None, :, :] * self.noop_bias

        # 6. Bucket Head (conditioned on source AND target)
        h_src = p_h[:, :, None, :]                           # (B, P, 1, d)
        h_tgt = p_h[:, None, :, :]                           # (B, 1, P, d)
        pair_h = jnp.concatenate([
            jnp.broadcast_to(h_src, (b, p, p, self.d_model)),
            jnp.broadcast_to(h_tgt, (b, p, p, self.d_model))
        ], axis=-1)                                          # (B, P, P, 2d)
        bucket_logits = self.bucket_head(pair_h)             # (B, P, P, BUCKETS)

        # 7. Value Head (from Global token)
        value = self.value_head(g_h).squeeze(-1)             # (B,)

        return PolicyOutput(
            target_logits=target_logits,
            bucket_logits=bucket_logits,
            value=value,
        )


def init_policy(
    rng: jax.Array,
    model: PlanetPolicy,
    example: dict[str, jnp.ndarray],
):
    """Initialize params from an example batch dict."""
    return model.init(rng, **example)
