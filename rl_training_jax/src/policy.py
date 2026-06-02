"""Transformer policy for Orbit Wars (JAX/Flax).

Consumes the per-entity feature dict produced by
`orbit_wars.features_jax.encode_observation` and emits, per source planet:

- target logits over MAX_PLANETS target slots (which planet to attack/reinforce)
- ship-bucket logits over BUCKET_COUNT discrete ship amounts
- a single scalar value (for the critic)

Design (kept small for Kaggle GPU inference):

    tokens = [planet_tokens]
    tokens = TransformerEncoder(d_model, n_heads, n_layers)(tokens, mask)
    target_logits = einsum("bsd, btd -> bst", planet_h, planet_h) / sqrt(d_model)
    bucket_logits = MLP(planet_h)
    value         = MLP(mean(planet_h))

Masks are applied:

- *source* mask (planet must be owned by the active player) — applied by the
  rollout/PPO code, not here. The model returns *raw* logits for every planet
  slot; the rollout layer masks invalid sources/targets/buckets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct


@struct.dataclass
class PolicyOutput:
    target_logits: jnp.ndarray   # (B, MAX_PLANETS, MAX_PLANETS)
    bucket_logits: jnp.ndarray   # (B, MAX_PLANETS, BUCKET_COUNT)
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

        # Self-attention with a key-padding mask. Flax expects a mask of
        # shape (B, num_heads, q_len, kv_len) where True means "keep".
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

        # Zero-out padding tokens to keep their representations clean for
        # downstream pooling/heads.
        tokens = tokens * kv_padding_mask[:, :, None].astype(tokens.dtype)
        return tokens


class PlanetPolicy(nn.Module):
    """Transformer over (planets) producing planet-action heads."""

    planet_count: int
    fleet_count: int  # Unused but kept for API compatibility with train_ppo setup
    bucket_count: int = 8
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ff_mult: int = 4
    noop_bias_init: float = 2.0  # Initial logit bias for self-target (NOOP)

    def setup(self) -> None:
        self.planet_in = nn.Dense(self.d_model)
        self.blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.ff_mult)
            for _ in range(self.num_layers)
        ]
        self.target_proj_q = nn.Dense(self.d_model)
        self.target_proj_k = nn.Dense(self.d_model)
        self.bucket_head = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(self.bucket_count)]
        )
        self.value_head = nn.Sequential(
            [nn.Dense(self.d_model), nn.gelu, nn.Dense(1)]
        )
        self.noop_bias = self.param(
            "noop_bias", nn.initializers.constant(self.noop_bias_init), ()
        )

    def __call__(
        self,
        planet_features: jnp.ndarray,    # (B, P, F_p)
        planet_mask: jnp.ndarray,        # (B, P) bool
    ) -> PolicyOutput:
        b, p, _ = planet_features.shape

        planet_tok = self.planet_in(planet_features)         # (B, P, d)

        tokens = planet_tok
        full_mask = planet_mask                              # (B, P)

        for block in self.blocks:
            tokens = block(tokens, full_mask)

        planet_h = tokens                                    # (B, P, d)

        # Target head: for each source planet, score every target planet via
        # scaled dot-product. Separate Q/K projections so source and target
        # representations can diverge.
        q = self.target_proj_q(planet_h)                     # (B, P, d)
        k = self.target_proj_k(planet_h)                     # (B, P, d)
        scale = jnp.float32(1.0 / jnp.sqrt(self.d_model))
        target_logits = jnp.einsum("bsd,btd->bst", q, k) * scale     # (B, P, P)

        # Add learnable NOOP bias to the diagonal (self-target)
        diag_mask = jnp.eye(p, dtype=target_logits.dtype)            # (P, P)
        target_logits = target_logits + diag_mask[None, :, :] * self.noop_bias

        # Bucket head: for every (source, target) pair, score ship buckets.
        # We concatenate source and target representations to allow the model
        # to decide ship amounts based on the specific target.
        h_src = planet_h[:, :, None, :]                       # (B, P, 1, d)
        h_tgt = planet_h[:, None, :, :]                       # (B, 1, P, d)
        
        # Broadcast and concatenate
        pair_h = jnp.concatenate([
            jnp.broadcast_to(h_src, (b, p, p, self.d_model)),
            jnp.broadcast_to(h_tgt, (b, p, p, self.d_model))
        ], axis=-1)                                          # (B, P, P, 2d)
        
        bucket_logits = self.bucket_head(pair_h)             # (B, P, P, BUCKETS)

        # Value head from mean pooling over valid planets.
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
