import jax
import jax.numpy as jnp
from rl_training_jax.src.ppo import joint_log_prob_and_entropy

N = 1
P = 96
BUCKETS = 8

# Let's check how fast entropy collapses to NOOP if the NOOP bias is heavily favored
# target_logits with a strong preference (e.g. 5.0) on diagonal
target_logits = jnp.zeros((N, P, P)) + jnp.eye(P)[None, :, :] * 5.0
bucket_logits = jnp.zeros((N, P, P, BUCKETS))
target_has_bucket = jnp.ones((N, P, P), dtype=jnp.bool_)
bucket_valid = jnp.ones((N, P, P, BUCKETS), dtype=jnp.bool_)
target_idx = jnp.zeros((N, P), dtype=jnp.int32)
bucket_idx = jnp.zeros((N, P), dtype=jnp.int32)
executed_mask = jnp.ones((N, P), dtype=jnp.bool_)

out = joint_log_prob_and_entropy(
    target_logits, bucket_logits, target_has_bucket, bucket_valid,
    target_idx, bucket_idx, executed_mask
)

print("Target Entropy (NOOP 5.0):", out["entropy_target"][0, 0])
print("Bucket Entropy:", out["entropy_bucket"][0, 0])
print("Total Entropy:", out["entropy_target"][0, 0] + out["entropy_bucket"][0, 0])
