import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset

state = reset(0, episode_steps=500)
grid = compose_action_grid(state, jnp.int32(0))

full_valid = np.asarray(grid["full_valid"])
source_valid = np.asarray(grid["source_valid"])

# How many targets are valid per source?
targets_per_source = np.sum(np.any(full_valid, axis=-1), axis=-1)
# How many buckets are valid per source/target pair?
buckets_per_pair = np.sum(full_valid, axis=-1)

print("Active sources:", np.sum(source_valid))
for i in range(len(source_valid)):
    if source_valid[i]:
        print(f"Source {i}: {targets_per_source[i]} targets.")
        # print max buckets for any target
        max_b = np.max(buckets_per_pair[i])
        print(f"  Max buckets for a target: {max_b}")

