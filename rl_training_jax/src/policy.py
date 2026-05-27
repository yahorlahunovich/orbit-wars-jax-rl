"""Transformer policy for Orbit Wars (JAX/Flax).

Consumes the per-entity feature dict produced by
`orbit_wars.features_jax.encode_observation` and emits, per source planet:

- target logits over MAX_PLANETS target slots (which planet to attack/reinforce)
- ship-bucket logits over BUCKET_COUNT discrete ship amounts
- a single scalar value (for the critic)

Design (kept small for Kaggle GPU inference):

    tokens = [global_token, planet_tokens, fleet_tokens]
    tokens = TransformerEncoder(d_model, n_heads, n_layers)(tokens, mask)
    target_logits = einsum("bsd, btd -> bst", planet_h, planet_h) / sqrt(d_model)
    bucket_logits = MLP(planet_h)
    value         = MLP(global_h)

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
    """Transformer over (global, planets, fleets) producing planet-action heads."""

    planet_count: int
    fleet_count: int
    bucket_count: int = 8
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ff_mult: int = 4

    def setup(self) -> None:
        # Token-type embedding: 0 = global, 1 = planet, 2 = fleet.
        self.type_emb = nn.Embed(3, self.d_model)
        self.planet_in = nn.Dense(self.d_model)
        self.fleet_in = nn.Dense(self.d_model)
        self.global_in = nn.Dense(self.d_model)
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

        planet_tok = self.planet_in(planet_features)         # (B, P, d)
        fleet_tok = self.fleet_in(fleet_features)            # (B, F, d)
        global_tok = self.global_in(global_features)         # (B, d)
        global_tok = global_tok[:, None, :]                  # (B, 1, d)

        type_idx = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.ones((p,), dtype=jnp.int32),
                jnp.full((f,), 2, dtype=jnp.int32),
            ],
            axis=0,
        )                                                    # (1+P+F,)
        type_vecs = self.type_emb(type_idx)                  # (1+P+F, d)

        tokens = jnp.concatenate([global_tok, planet_tok, fleet_tok], axis=1)   # (B, 1+P+F, d)
        tokens = tokens + type_vecs[None, :, :]
        full_mask = jnp.concatenate(
            [
                jnp.ones((b, 1), dtype=jnp.bool_),
                planet_mask,
                fleet_mask,
            ],
            axis=-1,
        )                                                    # (B, 1+P+F)

        for block in self.blocks:
            tokens = block(tokens, full_mask)

        global_h = tokens[:, 0, :]                           # (B, d)
        planet_h = tokens[:, 1 : 1 + p, :]                   # (B, P, d)

        # Target head: for each source planet, score every target planet via
        # scaled dot-product. Separate Q/K projections so source and target
        # representations can diverge.
        q = self.target_proj_q(planet_h)                     # (B, P, d)
        k = self.target_proj_k(planet_h)                     # (B, P, d)
        scale = jnp.float32(1.0 / jnp.sqrt(self.d_model))
        target_logits = jnp.einsum("bsd,btd->bst", q, k) * scale     # (B, P, P)

        # Bucket head: per source planet.
        bucket_logits = self.bucket_head(planet_h)           # (B, P, BUCKETS)

        # Value head from the global token.
        value = self.value_head(global_h).squeeze(-1)        # (B,)

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
