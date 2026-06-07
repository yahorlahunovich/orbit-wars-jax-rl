"""Evaluate JAX PPO agent against the sniper heuristic."""

import jax
import jax.numpy as jnp
from kaggle_environments import make

from src.policy import PlanetPolicy
from src.orbit_wars.features_jax import encode_observation
from src.orbit_wars.convert import observation_to_state
from src.orbit_wars.decode import compose_target_grid
from src.orbit_wars.rollout import sample_actions, pack_padded_actions

# Ensure sniper opponent exists
from heuristics.Simplified_Orbit_Wars_Agent_1245.submission import agent as sniper_agent

def run_evaluation(num_episodes=5):
    # Initialize the policy
    network = PlanetPolicy(
        planet_count=40,
        fleet_count=100,
        bucket_count=4,
        d_model=96,
        num_heads=4,
        num_layers=3,
    )
    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    
    init_feat = jnp.zeros((1, 40, 58))
    init_mask = jnp.zeros((1, 40), dtype=jnp.bool_)
    params = network.init(_rng, planet_features=init_feat, planet_mask=init_mask)

    # Fast forward: In a real scenario you would load trained params here.

    def agent(observation, configuration):
        nonlocal rng
        
        # 1. Convert Kaggle observation to JAX OrbitWarsState
        state = observation_to_state(
            observation,
            episode_seed=configuration.seed,
            ship_speed=6.0,
            episode_steps=500,
            done=False,
            rewards=(0.0, 0.0)
        )
        
        # 2. Encode features
        player = observation.player
        features = encode_observation(state, player=player)
        
        # 3. Add batch dim
        feat = jnp.expand_dims(features["planet_features"], 0)
        mask = jnp.expand_dims(features["planet_mask"], 0)
        
        # 4. Policy inference
        out = network.apply(params, planet_features=feat, planet_mask=mask)
        
        # 5. Decode
        phase1 = compose_target_grid(state, jnp.int32(0))
        # Add batch dim to phase1
        phase1 = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), phase1)
        state_batch = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)
        
        rng, _rng = jax.random.split(rng)
        sampled = sample_actions(
            rng=_rng,
            target_logits=out.target_logits,
            bucket_logits=out.bucket_logits,
            state=state_batch,
            phase1=phase1,
            deterministic=True, # Greedy eval
        )
        
        pids = jnp.expand_dims(state.planets[:, 0].astype(jnp.int32), 0)
        actions_packed, mask_packed, _ = pack_padded_actions(
            target_idx=sampled["target_idx"],
            bucket_idx=sampled["bucket_idx"],
            source_valid=sampled["source_valid"],
            from_ids=pids,
            angle=sampled["angle"],
            ship_counts=sampled["ship_counts"]
        )
        
        # 6. Convert to Kaggle actions
        moves = []
        for i in range(actions_packed.shape[1]):
            if mask_packed[0, i]:
                f = float(actions_packed[0, i, 0])
                a = float(actions_packed[0, i, 1])
                s = float(actions_packed[0, i, 2])
                moves.append([f, a, s])
                
        return moves

    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=False)
    
    wins = 0
    for i in range(num_episodes):
        steps = env.run([agent, sniper_agent])
        if steps[-1][0].reward > steps[-1][1].reward:
            wins += 1
            print(f"Episode {i+1}: WON")
        else:
            print(f"Episode {i+1}: LOST")
            
    print(f"Total Win Rate: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
