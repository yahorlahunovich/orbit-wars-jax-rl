import re

with open('rl_training_jax/src/train_ppo.py', 'r') as f:
    content = f.read()

# Update init_policy_params example input
content = re.sub(
    r'        "fleet_features": jnp.zeros\(\(1, MAX_FLEETS, FLEET_FEATURE_DIM\), jnp.float32\),\n        "fleet_mask": jnp.zeros\(\(1, MAX_FLEETS\), jnp.bool_\),\n        "global_features": jnp.zeros\(\(1, GLOBAL_FEATURE_DIM\), jnp.float32\),',
    '',
    content
)

# Update rollout_selfplay_factory
content = re.sub(
    r'            fleet_features=feats\["fleet_features"\],\n            fleet_mask=feats\["fleet_mask"\],\n            global_features=feats\["global_features"\],',
    '',
    content
)

# Update rollout_vs_heuristic_factory
content = re.sub(
    r'            fleet_features=feats_boot\["fleet_features"\],\n            fleet_mask=feats_boot\["fleet_mask"\],\n            global_features=feats_boot\["global_features"\],',
    '',
    content
)

content = re.sub(
    r'            fleet_features=learner_feats\["fleet_features"\],\n            fleet_mask=learner_feats\["fleet_mask"\],\n            global_features=learner_feats\["global_features"\],',
    '',
    content
)

# Update learner_record_from_samples
content = re.sub(
    r'        "fleet_features": learner_feats\["fleet_features"\],\n        "fleet_mask": learner_feats\["fleet_mask"\],\n        "global_features": learner_feats\["global_features"\],',
    '',
    content
)

# Update make_update_step 
content = re.sub(
    r'            fleet_features=sub\["fleet_features"\], fleet_mask=sub\["fleet_mask"\],\n            global_features=sub\["global_features"\],',
    '',
    content
)

with open('rl_training_jax/src/train_ppo.py', 'w') as f:
    f.write(content)
print("Updated train_ppo.py")
