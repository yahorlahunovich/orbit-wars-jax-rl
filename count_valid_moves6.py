import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset

state = reset(0, episode_steps=500)
grid = compose_action_grid(state, jnp.int32(0))

full_valid = np.asarray(grid["full_valid"])
source_valid = np.asarray(grid["source_valid"])
targets_per_source = np.sum(np.any(full_valid, axis=-1), axis=-1)

# Is target_valid_any actually false?
target_valid_any = np.any(full_valid, axis=(1, 2))

for i in range(len(source_valid)):
    if source_valid[i]:
        print(f"Source {i}: {targets_per_source[i]} valid targets in full_valid.")
        print(f"   But grid['target_valid_any'] says: {np.any(full_valid[i])}")
