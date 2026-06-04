"""Transformer policy for Orbit Wars (JAX/Flax).

Consumes the per-entity feature dict produced by
`orbit_wars.features_jax.encode_observation` and emits:

- target logits over MAX_PLANETS target slots (which planet to attack/reinforce)
- ship-bucket logits over (MAX_PLANETS, BUCKET_COUNT) conditional on target
- a single scalar value (for the critic)

Design:

    tokens = [planet_tokens]
    tokens = TransformerEncoder(d_model, n_heads, n_layers)(tokens, mask)
    target_logits = dot_product(planet_h, planet_h) + noop_bias
    bucket_logits = MLP(planet_h_src, planet_h_tgt)
    value         = MLP(mean(planet_h))
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
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )
        x_norm = nn.LayerNorm()(tokens)
        x_attn = attn(x_norm, x_norm, mask=mask)
        tokens = tokens + x_attn

        y_norm = nn.LayerNorm()(tokens)
        y = nn.Dense(
            self.d_model * self.ff_mult,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
        )(y_norm)
        y = nn.gelu(y)
        y = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(y)
        tokens = tokens + y

        tokens = tokens * kv_padding_mask[:, :, None].astype(tokens.dtype)
        return tokens


class PlanetPolicy(nn.Module):
    """Transformer over (planets) producing planet-action heads."""

    planet_count: int
    fleet_count: int  # API compatibility
    bucket_count: int = 8
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ff_mult: int = 4
    noop_bias_init: float = 1.0

    def setup(self) -> None:
        self.planet_in = nn.Dense(
            self.d_model, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
        )
        self.blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.ff_mult)
            for _ in range(self.num_layers)
        ]
        self.target_proj_q = nn.Dense(
            self.d_model, kernel_init=nn.initializers.orthogonal(0.01)
        )
        self.target_proj_k = nn.Dense(
            self.d_model, kernel_init=nn.initializers.orthogonal(0.01)
        )
        self.bucket_src = nn.Dense(
            self.bucket_count, kernel_init=nn.initializers.orthogonal(0.01)
        )
        self.bucket_tgt = nn.Dense(
            self.bucket_count, kernel_init=nn.initializers.orthogonal(0.01)
        )
        self.noop_bias = self.param(
            "noop_bias", nn.initializers.constant(self.noop_bias_init), (1,)
        )
        self.value_head = nn.Sequential(
            [
                nn.Dense(self.d_model, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0)),
            ]
        )

    def __call__(
        self,
        planet_features: jnp.ndarray,    # (B, P, F_p)
        planet_mask: jnp.ndarray,        # (B, P) bool
        **_kwargs,
    ) -> PolicyOutput:
        b, p, _ = planet_features.shape

        planet_tok = self.planet_in(planet_features)         # (B, P, d)

        tokens = planet_tok
        full_mask = planet_mask                              # (B, P)

        for block in self.blocks:
            tokens = block(tokens, full_mask)

        planet_h = tokens                                    # (B, P, d)

        # Target head
        q = self.target_proj_q(planet_h)                     # (B, P, d)
        k = self.target_proj_k(planet_h)                     # (B, P, d)
        scale = jnp.float32(1.0 / jnp.sqrt(self.d_model))
        target_logits = jnp.einsum("bsd,btd->bst", q, k) * scale     # (B, P, P)

        # Add noop bias to the diagonal (source == target)
        diag = jnp.eye(p, dtype=target_logits.dtype)         # (P, P)
        target_logits = target_logits + diag[None, :, :] * self.noop_bias

        # Bucket head: Factorized representation for fast O(P^2) assembly
        b_src = self.bucket_src(planet_h)                    # (B, P, BUCKETS)
        b_tgt = self.bucket_tgt(planet_h)                    # (B, P, BUCKETS)
        bucket_logits = b_src[:, :, None, :] + b_tgt[:, None, :, :]  # (B, P, P, BUCKETS)

        # Value head
        valid_count = jnp.maximum(jnp.sum(planet_mask, axis=1, keepdims=True), 1.0)
        mean_h = jnp.sum(planet_h, axis=1) / valid_count     # (B, d)
        value = self.value_head(mean_h).squeeze(-1)          # (B,)

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
