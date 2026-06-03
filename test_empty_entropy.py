import jax
import jax.numpy as jnp
from rl_training_jax.src.ppo import joint_log_prob_and_entropy

N = 1
P = 96
BUCKETS = 8

target_logits = jnp.zeros((N, P, P))
bucket_logits = jnp.zeros((N, P, P, BUCKETS))
target_has_bucket = jnp.ones((N, P, P), dtype=jnp.bool_)
bucket_valid = jnp.ones((N, P, P, BUCKETS), dtype=jnp.bool_)

# Planet 0 has NO valid targets
target_has_bucket = target_has_bucket.at[:, 0, :].set(False)

target_idx = jnp.zeros((N, P), dtype=jnp.int32)
bucket_idx = jnp.zeros((N, P), dtype=jnp.int32)

# It is executed (e.g. it was passed to PPO)
executed_mask = jnp.zeros((N, P), dtype=jnp.bool_).at[:, 0].set(True)

out = joint_log_prob_and_entropy(
    target_logits, bucket_logits, target_has_bucket, bucket_valid,
    target_idx, bucket_idx, executed_mask
)

print("Log Prob:", out["log_prob"][0, 0])
print("Target Entropy:", out["entropy_target"][0, 0])
print("Bucket Entropy:", out["entropy_bucket"][0, 0])
