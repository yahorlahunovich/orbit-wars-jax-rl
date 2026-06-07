import jax
import jax.numpy as jnp
import numpy as np
import yaml
import json
import flax.serialization
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "rl_training_jax/src"))

from orbit_wars import reset, encode_observation, compose_target_grid, MAX_PLANETS, MAX_FLEETS

import importlib.util
sub_policy = importlib.util.module_from_spec(importlib.util.spec_from_file_location("sub_policy", str(repo_root / "submission_jax/src/policy.py")))
sub_policy.__package__ = "sub_policy"
sys.modules["sub_policy"] = sub_policy
importlib.util.spec_from_file_location("sub_policy", str(repo_root / "submission_jax/src/policy.py")).loader.exec_module(sub_policy)
PlanetPolicy = sub_policy.PlanetPolicy

def load_submission_weights():
    weights_path = repo_root / "submission_jax/weights/policy.msgpack"
    config_path = repo_root / "submission_jax/weights/model_config.json"
    if not weights_path.exists(): return None, None
    cfg = json.loads(config_path.read_text())
    model = PlanetPolicy(planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS, d_model=cfg["d_model"], num_heads=cfg["num_heads"], num_layers=cfg["num_layers"], bucket_count=cfg["bucket_count"])
    raw = weights_path.read_bytes()
    init_params = model.init(jax.random.PRNGKey(0), planet_features=jnp.zeros((1, MAX_PLANETS, 12), jnp.float32), planet_mask=jnp.ones((1, MAX_PLANETS), jnp.bool_))
    params = flax.serialization.from_bytes(init_params, raw)
    return model, params

def _entropy(logits, mask):
    log_p = jax.nn.log_softmax(jnp.where(mask, logits, -1e9), axis=-1)
    p = jnp.exp(log_p) * mask.astype(jnp.float32)
    return -jnp.sum(p * log_p, axis=-1)

def analyze_entropy(model, params):
    state = reset(0, episode_steps=500)
    for i in range(5):
        states_batched = jax.tree_util.tree_map(lambda x: x[None, ...], state)
        feats = jax.vmap(encode_observation, in_axes=(0, None))(states_batched, jnp.int32(0))
        out = model.apply(params, planet_features=feats["planet_features"], planet_mask=feats["planet_mask"])
        phase1 = jax.vmap(compose_target_grid, in_axes=(0, None, 0, 0))(states_batched, jnp.int32(0), feats["incoming_me"], feats["incoming_enemy"])
        
        target_mask = phase1["target_mask"][0]
        source_valid = phase1["source_valid_any"][0]
        ent_target = _entropy(out.target_logits[0], target_mask)
        ent_bucket = _entropy(out.bucket_logits[0], jnp.ones_like(out.bucket_logits[0], dtype=jnp.bool_))
        total_ent = ent_target + ent_bucket
        
        target_idx = jnp.argmax(jnp.where(target_mask, out.target_logits[0], -1e9), axis=-1)
        is_noop = (target_idx == jnp.arange(MAX_PLANETS))
        
        noop_mask = source_valid & is_noop
        move_mask = source_valid & (~is_noop)
        
        print(f"\nStep {i}:")
        print(f"  Valid sources: {source_valid.sum()}")
        print(f"  NOOPs: {noop_mask.sum()}, Moves: {move_mask.sum()}")
        
        all_active_ent = jnp.sum(total_ent * source_valid) / jnp.maximum(source_valid.sum(), 1)
        move_only_ent = jnp.sum(total_ent * move_mask) / jnp.maximum(move_mask.sum(), 1)
        
        print(f"  Average Entropy (All Active): {all_active_ent:.4f}")
        print(f"  Average Entropy (Moves Only - what PPO sees): {move_only_ent:.4f}")
        
        if move_mask.any():
            idx = jnp.where(move_mask)[0][0]
            print(f"  Sample Move: Planet {idx} -> Target {target_idx[idx]} (Ent: {total_ent[idx]:.4f})")
        
        state = reset(i + 1, episode_steps=500)

if __name__ == "__main__":
    model, params = load_submission_weights()
    if model: analyze_entropy(model, params)
