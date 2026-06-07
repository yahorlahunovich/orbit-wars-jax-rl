"""Run a match of JAX PPO against the sniper heuristic and save to HTML."""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="ppo_jax/vs_sniper.html")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize the policy
    network = PlanetPolicy(
        planet_count=40,
        fleet_count=100,
        bucket_count=4,
        d_model=96,
        num_heads=4,
        num_layers=3,
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    
    init_feat = jnp.zeros((1, 40, 58))
    init_mask = jnp.zeros((1, 40), dtype=jnp.bool_)
    params = network.init(_rng, planet_features=init_feat, planet_mask=init_mask)

    def agent(observation, configuration):
        nonlocal rng
        
        state = observation_to_state(
            observation,
            episode_seed=configuration.seed,
            ship_speed=6.0,
            episode_steps=500,
            done=False,
            rewards=(0.0, 0.0)
        )
        
        player = observation.player
        features = encode_observation(state, player=player)
        
        feat = jnp.expand_dims(features["planet_features"], 0)
        mask = jnp.expand_dims(features["planet_mask"], 0)
        
        out = network.apply(params, planet_features=feat, planet_mask=mask)
        
        phase1 = compose_target_grid(state, jnp.int32(0))
        phase1 = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), phase1)
        state_batch = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)
        
        rng, _rng = jax.random.split(rng)
        sampled = sample_actions(
            rng=_rng,
            target_logits=out.target_logits,
            bucket_logits=out.bucket_logits,
            state=state_batch,
            phase1=phase1,
            deterministic=True,
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
        
        moves = []
        for i in range(actions_packed.shape[1]):
            if mask_packed[0, i]:
                f = float(actions_packed[0, i, 0])
                a = float(actions_packed[0, i, 1])
                s = float(actions_packed[0, i, 2])
                moves.append([f, a, s])
                
        return moves

    print("Playing match...")
    env = make("orbit_wars", configuration={"episodeSteps": 500, "randomSeed": args.seed}, debug=False)
    steps = env.run([agent, sniper_agent])
    
    print(f"Match finished. Saving HTML to {args.output}")
    with open(args.output, "w") as f:
        f.write(env.render(mode="html"))


if __name__ == "__main__":
    main()
