import jax
import jax.numpy as jnp
import equinox as eqx
from src.policy import GraphTransformerV9
from src.orbit_wars.features_jax import ObsBatch

def test_policy_forward():
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    
    # Instantiate GraphTransformerV9
    model = GraphTransformerV9(
        hidden_dim=64,
        n_layers=3,
        heads=4,
        n_ship_options=3,
        node_input_dim=21,
        edge_input_dim=14,
        edge_dim=16,
        n_policy_heads=1,
        key=k1
    )
    
    # Create fake single observation (representing 60 planets)
    obs = ObsBatch(
        node_features=jnp.zeros((60, 21), dtype=jnp.float32),
        edge_features=jnp.zeros((60, 60, 14), dtype=jnp.float32),
        future_sight=jnp.zeros((60, 32), dtype=jnp.float32),
        global_features=jnp.zeros((8,), dtype=jnp.float32),
        owned_nodes=jnp.full((60,), -1, dtype=jnp.int32).at[:5].set(jnp.arange(5)),
        edge_valid_mask=jnp.ones((60, 60, 3), dtype=jnp.bool_)
    )
    
    # Test encode pass
    node_h, edge_h = model.encode(
        obs.node_features,
        obs.edge_features,
        obs.future_sight,
        obs.global_features,
    )
    assert node_h.shape == (60, 64)
    assert edge_h.shape == (60, 60, 16)
    
    # Test policy head call
    head = model.policy_heads[0]
    send, target, frac = head(
        node_h,
        edge_h,
        obs.edge_features,
        obs.owned_nodes,
        obs.global_features,
    )
    # Target outputs should match shape (M, 60) and frac (M, 60, 3)
    # where M = shape of owned_nodes (60)
    assert send.shape == (60,)
    assert target.shape == (60, 60)
    assert frac.shape == (60, 60, 3)
    
    # Test value head call
    val = model.value_head(
        obs.node_features,
        obs.future_sight,
        obs.global_features,
        obs.edge_features,
    )
    assert val.shape == ()

def test_policy_backward_and_jit():
    rng = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(rng)
    
    model = GraphTransformerV9(
        hidden_dim=32,
        n_layers=2,
        heads=2,
        n_ship_options=3,
        node_input_dim=21,
        edge_input_dim=14,
        edge_dim=8,
        n_policy_heads=1,
        key=k1
    )
    
    obs = ObsBatch(
        node_features=jnp.zeros((60, 21), dtype=jnp.float32),
        edge_features=jnp.zeros((60, 60, 14), dtype=jnp.float32),
        future_sight=jnp.zeros((60, 32), dtype=jnp.float32),
        global_features=jnp.zeros((8,), dtype=jnp.float32),
        owned_nodes=jnp.full((60,), -1, dtype=jnp.int32).at[:5].set(jnp.arange(5)),
        edge_valid_mask=jnp.ones((60, 60, 3), dtype=jnp.bool_)
    )
    
    model_params, model_static = eqx.partition(model, eqx.is_array)
    
    @jax.jit
    def loss_fn(params, obs):
        m = eqx.combine(params, model_static)
        node_h, edge_h = m.encode(
            obs.node_features,
            obs.edge_features,
            obs.future_sight,
            obs.global_features,
        )
        head = m.policy_heads[0]
        send, target, frac = head(
            node_h,
            edge_h,
            obs.edge_features,
            obs.owned_nodes,
            obs.global_features,
        )
        val = m.value_head(
            obs.node_features,
            obs.future_sight,
            obs.global_features,
            obs.edge_features,
        )
        return jnp.mean(send) + jnp.mean(target) + jnp.mean(frac) + jnp.mean(val)
        
    loss_val, grads = jax.value_and_grad(loss_fn)(model_params, obs)
    assert jnp.isfinite(loss_val)
    
    # Check that gradients are computed for parameter leaf nodes
    flat_grads = jax.tree_util.tree_leaves(grads)
    assert len(flat_grads) > 0
    for g in flat_grads:
        assert jnp.all(jnp.isfinite(g))
