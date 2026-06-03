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

model = PlanetPolicy(planet_count=96, fleet_count=256)
rng = jax.random.PRNGKey(0)
dummy_feats = encode_batch(states, players)
params = model.init(rng, **dummy_feats)

@jax.jit
def run_encode(s, p):
    return encode_batch(s, p)

@jax.jit
def run_policy(p, f):
    return model.apply(p, **f)

@jax.jit
def run_grid(s, p):
    return jax.vmap(compose_action_grid, in_axes=(0, 0))(s, p)

@jax.jit
def run_sample_and_pack(r, t, b, g):
    s = sample_actions(r, t, b, g)
    a, m, e = pack_padded_actions(s["target_idx"], s["bucket_idx"], s["source_valid"], g)
    return a, m, e

@jax.jit
def run_step(s, a, m):
    return jax.vmap(step_jit)(s, a, jnp.zeros_like(a), m, jnp.zeros_like(m))

print("Compiling...")
feats = run_encode(states, players)
out = run_policy(params, feats)
grid = run_grid(states, players)
a, m, e = run_sample_and_pack(rng, out.target_logits, out.bucket_logits, grid)
next_states = run_step(states, a, m)
jax.block_until_ready(next_states)

def bench(name, fn, *args):
    # Warmup
    for _ in range(3):
        res = fn(*args)
    jax.block_until_ready(res)
    
    t0 = time.time()
    for _ in range(100):
        res = fn(*args)
    jax.block_until_ready(res)
    dur = time.time() - t0
    print(f"{name:>20}: {dur/100 * 1000:.2f} ms/call")

print("\nBenchmarking individual components (32 envs)...")
bench("Encode", run_encode, states, players)
bench("Policy Forward", run_policy, params, feats)
bench("Compose Grid", run_grid, states, players)
bench("Sample & Pack", run_sample_and_pack, rng, out.target_logits, out.bucket_logits, grid)
bench("Env Step", run_step, states, a, m)
