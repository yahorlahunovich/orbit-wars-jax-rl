with open('rl_training_jax/tests/test_decode.py', 'r') as f:
    content = f.read()

content = content.replace(
    'sc_small = ship_counts_for_buckets(jnp.float32(50.0), jnp.float32(10.0))',
    'sc_small = ship_counts_for_buckets(jnp.float32(50.0), jnp.float32(10.0), jnp.float32(0.0), jnp.float32(0.0))'
)

content = content.replace(
    'sc_large = ship_counts_for_buckets(jnp.float32(100.0), jnp.float32(10.0))',
    'sc_large = ship_counts_for_buckets(jnp.float32(100.0), jnp.float32(10.0), jnp.float32(0.0), jnp.float32(0.0))'
)

content = content.replace(
    'sc = ship_counts_for_buckets(jnp.float32(20.0), jnp.float32(2.0))',
    'sc = ship_counts_for_buckets(jnp.float32(20.0), jnp.float32(2.0), jnp.float32(0.0), jnp.float32(0.0))'
)

content = content.replace(
    'sc = ship_counts_for_buckets(jnp.float32(5.0), jnp.float32(100.0))',
    'sc = ship_counts_for_buckets(jnp.float32(5.0), jnp.float32(100.0), jnp.float32(0.0), jnp.float32(0.0))'
)

with open('rl_training_jax/tests/test_decode.py', 'w') as f:
    f.write(content)
