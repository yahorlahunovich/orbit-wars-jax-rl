import jax
import jax.numpy as jnp
from rl_training_jax.src.ppo import joint_log_prob_and_entropy

N = 1
P = 96
BUCKETS = 8

# Baseline: Uniform logits
target_logits = jnp.zeros((N, P, P))
bucket_logits = jnp.zeros((N, P, P, BUCKETS))

# Let's see what happens if a source only has 1 valid target and 1 valid bucket.
thb = jnp.zeros((N, P, P), dtype=jnp.bool_).at[:, :, 0].set(True)
bv = jnp.zeros((N, P, P, BUCKETS), dtype=jnp.bool_).at[:, :, :, 0].set(True)

t_idx = jnp.zeros((N, P), dtype=jnp.int32)
b_idx = jnp.zeros((N, P), dtype=jnp.int32)
mask = jnp.ones((N, P), dtype=jnp.bool_)

out = joint_log_prob_and_entropy(target_logits, bucket_logits, thb, bv, t_idx, b_idx, mask)

print("Target:", out["entropy_target"][0, 0])
print("Bucket:", out["entropy_bucket"][0, 0])

# What if 95 out of 96 planets are masked out?
mask2 = jnp.zeros((N, P), dtype=jnp.bool_).at[:, 0].set(True)
out2 = joint_log_prob_and_entropy(target_logits, bucket_logits, thb, bv, t_idx, b_idx, mask2)
print("\nTarget (95 masked):", jnp.sum(out2["entropy_target"]))
