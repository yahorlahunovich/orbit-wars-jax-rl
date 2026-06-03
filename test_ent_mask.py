import jax
import jax.numpy as jnp
from rl_training_jax.src.ppo import joint_log_prob_and_entropy

N = 1
P = 96
BUCKETS = 8

# Baseline: Uniform logits
target_logits = jnp.zeros((N, P, P))
bucket_logits = jnp.zeros((N, P, P, BUCKETS))

# Scenario A: 2 valid targets, 2 valid buckets
thb_a = jnp.zeros((N, P, P), dtype=jnp.bool_).at[:, :, :2].set(True)
bv_a = jnp.zeros((N, P, P, BUCKETS), dtype=jnp.bool_).at[:, :, :, :2].set(True)

# Scenario B: 5 valid targets, 8 valid buckets
thb_b = jnp.zeros((N, P, P), dtype=jnp.bool_).at[:, :, :5].set(True)
bv_b = jnp.zeros((N, P, P, BUCKETS), dtype=jnp.bool_).at[:, :, :, :8].set(True)

t_idx = jnp.zeros((N, P), dtype=jnp.int32)
b_idx = jnp.zeros((N, P), dtype=jnp.int32)
mask = jnp.ones((N, P), dtype=jnp.bool_)

out_a = joint_log_prob_and_entropy(target_logits, bucket_logits, thb_a, bv_a, t_idx, b_idx, mask)
out_b = joint_log_prob_and_entropy(target_logits, bucket_logits, thb_b, bv_b, t_idx, b_idx, mask)

print("Target A:", out_a["entropy_target"][0, 0])
print("Bucket A:", out_a["entropy_bucket"][0, 0])
print("Target B:", out_b["entropy_target"][0, 0])
print("Bucket B:", out_b["entropy_bucket"][0, 0])
