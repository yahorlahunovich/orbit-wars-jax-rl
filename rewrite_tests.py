import os
import glob
import re

files = glob.glob('rl_training_jax/**/*.py', recursive=True) + glob.glob('submission_jax/**/*.py', recursive=True)

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    orig_content = content

    # Replace imports
    content = content.replace('FLEET_FEATURE_DIM, ', '')
    content = content.replace('GLOBAL_FEATURE_DIM, ', '')
    content = content.replace(', FLEET_FEATURE_DIM', '')
    content = content.replace(', GLOBAL_FEATURE_DIM', '')
    
    # Remove from dicts
    content = re.sub(r'\s*"fleet_features":.*?,?\n', '\n', content)
    content = re.sub(r'\s*"fleet_mask":.*?,?\n', '\n', content)
    content = re.sub(r'\s*"global_features":.*?,?\n', '\n', content)
    
    # Remove from __call__ or similar
    content = re.sub(r'\s*fleet_features=.*?,?\n', '\n', content)
    content = re.sub(r'\s*fleet_mask=.*?,?\n', '\n', content)
    content = re.sub(r'\s*global_features=.*?,?\n', '\n', content)
    
    # Remove feature dims
    content = re.sub(r'\s*"fleet_feature_dim":.*?,?\n', '\n', content)
    content = re.sub(r'\s*"global_feature_dim":.*?,?\n', '\n', content)

    if orig_content != content:
        with open(file, 'w') as f:
            f.write(content)
        print(f"Updated {file}")
