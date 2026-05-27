
PPO approach.

1. JAX model + environment

2. Parse state into observations. put as much inductive bias as possible(can be up to 1000 features).
    - Gloval state
    - Feature state
    - More inductive bias is better.
    - Predictions 
Bad:

planet_ships=50

Better:

planet_ships=50

incoming_enemy=20
incoming_friendly=10

distance_to_home=15

nearest_enemy_distance=8

planet_value=
production/(ships+1)

reachable_without_sun=1

3. If we have explained variance >0.9 training will be good.
4. Gamma is close to 1 (0.9999 or even 1)
5. Transformer architecture(not so big to start): head for every output, value function, actions, fleets. We need to create planet embeddings 
    Just example: planet_embedding = [
    x,
    y,
    owner,
    ships,
    production,
    radius,
    is_comet,
    distance_to_home,
    incoming_enemy,
    incoming_friendly
]

Then:

Transformer(
    planets + fleets + comets
)

6. Actions are pretty simple: for every our planet choose one of our planets(even current) or enemy planets and decide how many fleet to send there(may be in bins). Angle is calculated by geometry functions.
7. Main metrics: clip fraction, kl, explained variance
8. Rollout steps = 32
9. Ideally to have batch size of 2048 per minibatch
10. Self-play training
11. Log steps per second speed. Our goal is to have up to 10 000 steps per second.
12. +1 / −1 /0 rewards for win/loss/draw only
13. cosine decay is needed
