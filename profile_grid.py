import jax
import jax.numpy as jnp
import time
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.reset import reset

num_envs = 32
state = reset(0, episode_steps=500)
states = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], num_envs, axis=0), state)
players = jnp.zeros(num_envs, dtype=jnp.int32)

@jax.jit
def run_grid(s, p):
    return jax.vmap(compose_action_grid, in_axes=(0, 0))(s, p)

from functools import partial

@jax.jit
def run_grid_no_planet_block(s, p):
    return jax.vmap(partial(compose_action_grid, enable_planet_block=False), in_axes=(0, 0))(s, p)

@jax.jit
def run_grid_no_intercept(s, p):
    return jax.vmap(partial(compose_action_grid, intercept_iterations=0), in_axes=(0, 0))(s, p)

out = run_grid(states, players)
jax.block_until_ready(out)

def bench(name, fn):
    t0 = time.time()
    for _ in range(100):
        res = fn(states, players)
    jax.block_until_ready(res)
    print(f"{name:>20}: {(time.time() - t0)/100 * 1000:.2f} ms/call")

bench("Normal", run_grid)
out2 = run_grid_no_planet_block(states, players)
jax.block_until_ready(out2)
bench("No Planet Block", run_grid_no_planet_block)
out3 = run_grid_no_intercept(states, players)
jax.block_until_ready(out3)
bench("No Intercept Iter", run_grid_no_intercept)

