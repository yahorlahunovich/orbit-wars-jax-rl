import jax.numpy as jnp
import numpy as np
from rl_training_jax.src.orbit_wars.geometry import solve_intercept_with_wait, get_arrival_turns

fx, fy, fsr = 29.280804, 8.687617, 1.5
tx, ty, ttr = 6.4874873, 78.19432, 1.5
orbiting = False
ships = 102.0
omega = 0.03547847
max_speed = 6.0

jix, jiy, jtt, _jb = solve_intercept_with_wait(
    jnp.float32(fx), jnp.float32(fy), jnp.float32(fsr),
    jnp.int32(-1),
    jnp.float32(tx), jnp.float32(ty), jnp.float32(ttr),
    jnp.bool_(orbiting), jnp.float32(ships),
    jnp.float32(omega), jnp.float32(max_speed),
    n_iter=6
)

print(f"JAX: x={jix}, y={jiy}, t={jtt}")
print(f"Original tx={tx}, ty={ty}")
