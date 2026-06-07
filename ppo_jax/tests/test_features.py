import jax
import jax.numpy as jnp
from src.orbit_wars.features_jax import extract_obs_v9_jax, ObsBatch
from src.env import OrbitWarsPureJaxEnv

def test_extract_obs_shapes():
    rng = jax.random.PRNGKey(0)
    env = OrbitWarsPureJaxEnv(episode_steps=10)
    
    # Reset environment to get a state
    obs, state = env.reset(rng)
    
    # Verify observation types and shapes
    assert isinstance(obs, ObsBatch)
    assert obs.node_features.shape == (60, 21)
    assert obs.edge_features.shape == (60, 60, 14)
    assert obs.future_sight.shape == (60, 32)
    assert obs.global_features.shape == (8,)
    assert obs.owned_nodes.shape == (60,)
    assert obs.edge_valid_mask.shape == (60, 60, 3)

def test_extract_obs_jit():
    rng = jax.random.PRNGKey(42)
    env = OrbitWarsPureJaxEnv(episode_steps=10)
    _, state = env.reset(rng)
    
    @jax.jit
    def compile_extract(st):
        return extract_obs_v9_jax(st, player_id=0)
        
    obs = compile_extract(state)
    assert obs.node_features.shape == (60, 21)
    assert obs.edge_features.shape == (60, 60, 14)

def test_extract_obs_vmap():
    rng = jax.random.PRNGKey(123)
    env = OrbitWarsPureJaxEnv(episode_steps=10)
    
    # Reset multiple environments
    reset_rngs = jax.random.split(rng, 3)
    obs_batch, state_batch = jax.vmap(env.reset)(reset_rngs)
    
    assert obs_batch.node_features.shape == (3, 60, 21)
    assert obs_batch.edge_features.shape == (3, 60, 60, 14)
    assert obs_batch.future_sight.shape == (3, 60, 32)
    assert obs_batch.global_features.shape == (3, 8)
    assert obs_batch.owned_nodes.shape == (3, 60)
    assert obs_batch.edge_valid_mask.shape == (3, 60, 60, 3)
