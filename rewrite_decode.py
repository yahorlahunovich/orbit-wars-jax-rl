import re

with open('rl_training_jax/src/orbit_wars/decode.py', 'r') as f:
    content = f.read()

# Replace the docstring and coefficients for buckets
new_docstring = """\"\"\"Pure-JAX geometry decoder for Orbit Wars actions.

Given a chosen `(source_planet, target_planet, bucket)` triple, produce a
legal `[from_id, angle, num_ships]` action row plus validity flags. All
operations are vectorized and vmap-friendly — Phase 4 (rollout) will broadcast
over batch × source.

Ship-bucket scheme (BUCKET_COUNT = 8):

    0  25%  of source ships    (min 4 ships)
    1  50%  of source ships
    2  75%  of source ships
    3 100%  of source ships    (all-in)
    4  target_ships + 1        (minimal capture)
    5  target_ships + 50% src  (capture with reserve)
    6  target_ships + inc_enemy - inc_allied + 1 (smart capture minimal)
    7  target_ships + inc_enemy - inc_allied + 25% src (smart capture reserve)

Buckets are masked invalid when the computed ship count is <= 0 or exceeds the
source planet's current ship count.

A move is masked invalid when:

- source planet is not active or not owned by `player`;
- target planet is not active;
- the chosen ship count is 0 or > source ships;
- the straight-line path from source to target crosses the sun;
- the source and target are the same (degenerate self-launch).
\"\"\""""

content = re.sub(r'\"\"\"Pure-JAX geometry decoder for Orbit Wars actions\..*?\"\"\"', new_docstring, content, flags=re.DOTALL)

# Remove the old _SRC_FRAC, _TGT_FRAC, _PLUS, _MIN
content = re.sub(r'# Per-bucket coefficients.*?\n_MIN = .*?\n', '', content, flags=re.DOTALL)

# Update ship_counts_for_buckets
new_ship_counts = """def ship_counts_for_buckets(
    source_ships: jnp.ndarray, target_ships: jnp.ndarray, incoming_me: jnp.ndarray, incoming_enemy: jnp.ndarray
) -> jnp.ndarray:
    \"\"\"Return integer-valued ship counts for every bucket index.

    Inputs broadcast against each other. Output shape = broadcasted shape +
    `(BUCKET_COUNT,)`.
    \"\"\"
    src = source_ships[..., None]
    tgt = target_ships[..., None]
    inc_me = incoming_me[..., None]
    inc_en = incoming_enemy[..., None]
    
    b0 = src * 0.25
    b1 = src * 0.50
    b2 = src * 0.75
    b3 = src * 1.00
    b4 = tgt + 1.0
    b5 = tgt + src * 0.50
    b6 = jnp.maximum(0.0, tgt + inc_en - inc_me) + 1.0
    b7 = jnp.maximum(0.0, tgt + inc_en - inc_me) + src * 0.25
    
    raw = jnp.concatenate([b0, b1, b2, b3, b4, b5, b6, b7], axis=-1)
    raw = jnp.maximum(raw, jnp.float32(MIN_LAUNCH_SHIPS))
    # Floor to int while keeping floats (the env stores ships as float32 ints).
    return jnp.floor(raw)"""

content = re.sub(r'def ship_counts_for_buckets\(.*?return jnp\.floor\(raw\)', new_ship_counts, content, flags=re.DOTALL)

# Update compose_action_grid to use _fleet_projections
new_compose_action_grid = """def compose_action_grid(
    state: OrbitWarsState,
    player: jnp.int32 | int,
) -> dict[str, jnp.ndarray]:
    \"\"\"Pre-compute everything the policy/rollout needs about every
    (source, target, bucket) triple in a single state.

    Returns a dict with all (P_src, P_tgt) / (P_src, P_tgt, BUCKETS) shaped
    arrays:

        source_valid   (P,)             bool          source planet owned by player
        target_valid   (P,)             bool          target planet is active
        angle          (P, P, BUCKETS)  float32       safe intercept aim per bucket
        sun_blocks     (P, P, BUCKETS)  bool          launch→aim crosses sun
        planet_blocks  (P, P, BUCKETS)  bool          another planet blocks path
        self_target    (P, P)           bool          true on the diagonal
        target_valid_pair (P, P)        bool          target_valid AND not self
        ship_counts    (P, P, BUCKETS)  float32       per-bucket ship count to send
        bucket_valid   (P, P, BUCKETS)  bool          ship count fits source's reserve
        pair_valid     (P, P)           bool          source_valid AND target_valid_pair
        full_valid     (P, P, BUCKETS)  bool          pair & bucket & !sun & !planet block
        from_ids       (P,)             float32       planet id per source slot
    \"\"\"
    planets = state.planets
    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    player_f = jnp.float32(player)
    source_valid = active & (owner == player_f)
    target_valid = active

    x = planets[:, 2]
    y = planets[:, 3]
    radius = planets[:, 4]
    ships = planets[:, 5]

    tgt_orbiting = is_orbiting_planet(x, y, radius)  # (P,)

    from .features_jax import _fleet_projections
    incoming_me, incoming_enemy, _, _ = _fleet_projections(state, player_f)

    src_ships_grid = ships[:, None]                  # (P, 1)
    tgt_ships_grid = ships[None, :]                  # (1, P)
    inc_me_grid = incoming_me[None, :]               # (1, P)
    inc_en_grid = incoming_enemy[None, :]            # (1, P)
    ship_counts = ship_counts_for_buckets(src_ships_grid, tgt_ships_grid, inc_me_grid, inc_en_grid)  # (P, P, B)"""

content = re.sub(r'def compose_action_grid\(.*?ship_counts = ship_counts_for_buckets\(src_ships_grid, tgt_ships_grid\)  # \(P, P, B\)', new_compose_action_grid, content, flags=re.DOTALL)

with open('rl_training_jax/src/orbit_wars/decode.py', 'w') as f:
    f.write(content)
print("Updated decode.py")