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
    targets_per_source = np.sum(np.any(full_valid, axis=-1), axis=-1)
    
    if np.sum(targets_per_source) == 0:
        print(f"Step {step_idx}: ZERO valid targets! Why?")
        # Let's inspect the masks that make up full_valid
        pair_valid = np.asarray(grid["pair_valid"])
        bucket_valid = np.asarray(grid["bucket_valid"])
        sun_blocks = np.asarray(grid["sun_blocks"])
        planet_blocks = np.asarray(grid["planet_blocks"])
        
        print(f"pair_valid sum: {np.sum(pair_valid)}")
        print(f"bucket_valid sum: {np.sum(bucket_valid)}")
        print(f"sun_blocks sum: {np.sum(sun_blocks)}")
        print(f"planet_blocks sum: {np.sum(planet_blocks)}")
        
        for i in range(len(source_valid)):
            if source_valid[i]:
                # check buckets for this source
                b_val = bucket_valid[i]
                print(f"Source {i} buckets valid sum: {np.sum(b_val)}")
                if np.sum(b_val) == 0:
                    ships = state.planets[i, 5]
                    print(f"Source {i} has {ships} ships.")
        break
