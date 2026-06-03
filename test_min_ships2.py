import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.decode import ship_counts_for_buckets, MIN_LAUNCH_SHIPS, bucket_validity_mask

src = jnp.array([4.0])
tgt = jnp.array([10.0])
inc_me = jnp.array([0.0])
inc_en = jnp.array([0.0])

ships = ship_counts_for_buckets(src, tgt, inc_me, inc_en)
valid = bucket_validity_mask(ships, src)
print("Source ships = 4.0")
print(f"Bucket ship counts: {ships}")
print(f"Bucket valid: {valid}")
