import jax
import jax.numpy as jnp
import time
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset

state = reset(0, episode_steps=500)
# vmap to 32 envs
states = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], 32, axis=0), state)
players = jnp.zeros(32, dtype=jnp.int32)

@jax.jit
def run_grid(states, players):
    return jax.vmap(compose_action_grid, in_axes=(0, 0))(states, players)

print("Compiling...")
t0 = time.time()
out = run_grid(states, players)
jax.block_until_ready(out)
print(f"Compile + first run: {time.time() - t0:.2f}s")

print("Benchmarking...")
t0 = time.time()
for _ in range(10):
    out = run_grid(states, players)
jax.block_until_ready(out)
dur = time.time() - t0
print(f"10 runs: {dur:.2f}s -> {10 * 32 / dur:.2f} envs/s")
