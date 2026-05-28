import re

with open('rl_training_jax/tests/test_features_jax.py', 'r') as f:
    content = f.read()

# Fix indexing
content = content.replace('ship_rank = pf[:, 22]', 'ship_rank = pf[:, 24]')
content = content.replace('prod_rank = pf[:, 23]', 'prod_rank = pf[:, 25]')
content = content.replace('remaining = pf[is_comet, 30]', 'remaining = pf[is_comet, 32]')
content = content.replace('assert out["fleet_features"].shape == (MAX_FLEETS)', '# no fleet_features')
content = content.replace('fleet_features = np.asarray(out["fleet_features"])', '# fleet_features removed')
content = content.replace('assert np.all(fleet_features[~mask] == 0.0)', '')
content = content.replace('assert jnp.all(jnp.isfinite(out["fleet_features"]))', '')

# Remove global features test
content = re.sub(r'def test_global_lead_signs_flip\(\):.*?\n\n\n', '\n', content, flags=re.DOTALL)

with open('rl_training_jax/tests/test_features_jax.py', 'w') as f:
    f.write(content)
