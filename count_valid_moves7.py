import jax
import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset
from rl_training_jax.src.orbit_wars.step import step

state = reset(0, episode_steps=500)

for step_idx in range(50):
    grid = compose_action_grid(state, jnp.int32(0))
    full = np.asarray(grid["full_valid"])
    idxs = np.argwhere(full)
    
    if len(idxs) > 0:
        # Pick a random valid move
        s_idx, t_idx, b_idx = idxs[0]
        from_id = float(grid["from_ids"][s_idx])
        angle = float(grid["angle"][s_idx, t_idx, b_idx])
        ships = int(grid["ship_counts"][s_idx, t_idx, b_idx])
        action = [[from_id, angle, ships]]
    else:
        action = []
        
    state = step(state, [action, []])

    grid = compose_action_grid(state, jnp.int32(0))
    full_valid = np.asarray(grid["full_valid"])
    source_valid = np.asarray(grid["source_valid"])
    
    # Are there active sources that have no valid targets?
    targets_per_source = np.sum(np.any(full_valid, axis=-1), axis=-1)
    invalid_sources = source_valid & (targets_per_source == 0)
    
    if np.any(invalid_sources):
        print(f"Step {step_idx}: Found {np.sum(invalid_sources)} active sources with ZERO valid targets.")
        
