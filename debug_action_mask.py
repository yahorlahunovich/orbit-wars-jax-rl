import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset

state = reset(0, episode_steps=500)
grid = compose_action_grid(state, jnp.int32(0))

full_valid = np.asarray(grid["full_valid"])
source_valid = np.asarray(grid["source_valid"])

for i in range(len(source_valid)):
    if source_valid[i]:
        targets_valid = np.sum(full_valid[i], axis=-1)
        if np.sum(targets_valid) == 0:
            print(f"Source {i} has 0 valid targets.")
            print("Why?")
            print(f"  Self target allowed? {grid['self_target'][i, i]}")
            print(f"  Target valid pair? {grid['target_valid_pair'][0, i]}")
            print(f"  Pair valid? {grid['pair_valid'][i, i]}")
            print(f"  Sun blocks? {grid['sun_blocks'][i, i]}")
            print(f"  Planet blocks (vs self)? {np.any(grid['planet_blocks'][i, i])}")
            print(f"  Bucket valid (vs self)? {grid['bucket_valid'][i, i]}")

