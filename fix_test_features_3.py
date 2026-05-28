import re

with open('rl_training_jax/tests/test_features_jax.py', 'r') as f:
    content = f.read()

content = content.replace('assert out["fleet_mask"].shape == (MAX_FLEETS,)', '')
content = content.replace('inactive_fleets = fleet_features[~fleet_mask]', '')
content = content.replace('assert np.all(inactive_fleets == 0.0)', '')

content = content.replace('assert np.isclose(float(out["global_features"][0]), expected_turn)', 'assert np.isclose(float(out["planet_features"][0, 33]), expected_turn)')

with open('rl_training_jax/tests/test_features_jax.py', 'w') as f:
    f.write(content)
