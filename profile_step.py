import jax
import jax.numpy as jnp
import time
from rl_training_jax.src.orbit_wars.decode import compose_action_grid
from rl_training_jax.src.orbit_wars.features_jax import encode_batch
from rl_training_jax.src.orbit_wars.reset import reset
from rl_training_jax.src.orbit_wars.rollout import sample_actions, pack_padded_actions
from rl_training_jax.src.policy import PlanetPolicy
from rl_training_jax.src.orbit_wars.step import step_jit

num_envs = 32

state = reset(0, episode_steps=500)
states = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], num_envs, axis=0), state)
players = jnp.zeros(num_envs, dtype=jnp.int32)

@jax.jit
def run_step(s, p):
    # This matches exactly what the self-play loop does for one env step
    grid0 = jax.vmap(compose_action_grid, in_axes=(0, None))(s, jnp.int32(0))
    grid1 = jax.vmap(compose_action_grid, in_axes=(0, None))(s, jnp.int32(1))
    return grid0, grid1

print("Compiling...")
out = run_step(states, players)
jax.block_until_ready(out)

print("Benchmarking...")
t0 = time.time()
for _ in range(10):
    res = run_step(states, players)
jax.block_until_ready(res)
dur = time.time() - t0
print(f"Double grid generation (32 envs): {dur/10 * 1000:.2f} ms")
