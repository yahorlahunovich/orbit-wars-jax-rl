import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.policy import PlanetPolicy
from rl_training_jax.src.ppo import ppo_loss_fn
from orbit_wars import MAX_PLANETS, MAX_FLEETS, PLANET_FEATURE_DIM

def test_normalization_impact():
    # 1. Setup dummy model and data
    rng = jax.random.PRNGKey(42)
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS, d_model=32)
    
    # Simulate a typical batch of 128 env-steps
    N = 128
    P = MAX_PLANETS
    example = {
        "planet_features": jax.random.normal(rng, (N, P, PLANET_FEATURE_DIM)),
        "planet_mask": jnp.ones((N, P), dtype=jnp.bool_),
    }
    params = model.init(rng, **example)

    # 2. Create raw advantages (typical for Orbit Wars: small signals)
    # Most advantages are very close to 0 because the critic is good.
    raw_advantages = jax.random.normal(rng, (N,)) * 0.01 
    
    batch = {
        **example,
        "target_idx": jnp.zeros((N, P), dtype=jnp.int32),
        "bucket_idx": jnp.zeros((N, P), dtype=jnp.int32),
        "source_valid": jnp.ones((N, P), dtype=jnp.bool_),
        "old_log_prob": jnp.zeros((N, P)),
        "target_has_bucket": jnp.ones((N, P, P), dtype=jnp.bool_),
        "bucket_valid": jnp.ones((N, P, P, 8), dtype=jnp.bool_),
        "advantages": raw_advantages,
        "returns": jnp.zeros((N,)),
    }

    # 3. Define a "Raw" loss function (manually stripping normalization for this test)
    def raw_loss_fn(p, b):
        # This is a temporary re-implementation of ppo_loss_fn WITHOUT normalization
        out = model.apply(p, **{"planet_features": b["planet_features"], "planet_mask": b["planet_mask"]})
        # ... stripped down logic ...
        # (For the sake of the test, let's just look at policy gradient magnitude)
        logits = out.target_logits
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Gradient ~ log_prob * advantages
        return -jnp.mean(log_probs * b["advantages"][:, None, None])

    # 4. Define the "Normalized" loss function
    def normalized_loss_fn(p, b):
        adv = b["advantages"]
        adv_norm = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        out = model.apply(p, **{"planet_features": b["planet_features"], "planet_mask": b["planet_mask"]})
        logits = out.target_logits
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs * adv_norm[:, None, None])

    # 5. Compute Gradients
    grad_raw = jax.grad(raw_loss_fn)(params, batch)
    grad_norm = jax.grad(normalized_loss_fn)(params, batch)

    # 6. Calculate total gradient magnitude (norm)
    def total_norm(g):
        return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))

    norm_val_raw = total_norm(grad_raw)
    norm_val_norm = total_norm(grad_norm)

    print(f"--- Gradient Signal Strength ---")
    print(f"Raw Advantage Norm:        {norm_val_raw:.8f}")
    print(f"Normalized Advantage Norm: {norm_val_norm:.8f}")
    print(f"Signal Boost:              {norm_val_norm / norm_val_raw:.1f}x")

if __name__ == "__main__":
    test_normalization_impact()
