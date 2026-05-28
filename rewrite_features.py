import re

with open('rl_training_jax/src/orbit_wars/features_jax.py', 'r') as f:
    content = f.read()

# Define the new encode_observation function
new_encode = """def encode_observation(
    state: OrbitWarsState,
    player: jnp.int32 | int,
) -> dict[str, jnp.ndarray]:
    \"\"\"Encode a single (unbatched) state into entity features for `player`.

    Returns a dict with keys:
        planet_features (P, F_PLANET) float32
        planet_mask     (P,)           bool

    Apply `jax.vmap(encode_observation, in_axes=(0, None))` for batched envs.
    \"\"\"
    player_f = jnp.float32(player)
    planets = state.planets
    fleets = state.fleets

    active = planets[:, 7] > 0.0
    owner = planets[:, 1]
    owner_is_me = active & (owner == player_f)
    owner_is_enemy = active & (owner >= 0.0) & (owner != player_f)
    owner_is_neutral = active & (owner < 0.0)

    ships = planets[:, 5]
    production = planets[:, 6]
    radius = planets[:, 4]
    x = planets[:, 2]
    y = planets[:, 3]

    dx_c = (x - CENTER) / DIST_DENOM
    dy_c = (y - CENTER) / DIST_DENOM
    dist_c = jnp.sqrt(dx_c * dx_c + dy_c * dy_c)

    is_orbiting = _is_orbiting_per_planet(state) & active
    is_comet = _is_comet_per_planet(state) & active

    init = state.initial_planets
    dx0 = init[:, 2] - CENTER
    dy0 = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    initial_angle = jnp.arctan2(dy0, dx0)
    current_angle = initial_angle + state.angular_velocity * state.step.astype(jnp.float32)
    orbit_angle_sin = jnp.where(is_orbiting, jnp.sin(current_angle), 0.0)
    orbit_angle_cos = jnp.where(is_orbiting, jnp.cos(current_angle), 0.0)
    orbit_r_norm = jnp.where(is_orbiting, orbit_r / DIST_DENOM, 0.0)

    incoming_me, incoming_enemy, eta_me_norm, eta_enemy_norm = _fleet_projections(state, player_f)

    nearest_enemy_d_raw = _nearest_distance_to_subset(planets, owner_is_enemy)
    nearest_friend_d_raw = _nearest_distance_to_subset(planets, owner_is_me)
    nearest_neutral_d_raw = _nearest_distance_to_subset(planets, owner_is_neutral)
    nearest_enemy_d = nearest_enemy_d_raw / DIST_DENOM
    nearest_friend_d = nearest_friend_d_raw / DIST_DENOM

    max_speed = state.ship_speed
    time_norm = jnp.float32(100.0)
    time_to_nearest_enemy = jnp.where(active, nearest_enemy_d_raw / max_speed / time_norm, 0.0)
    time_to_nearest_neutral = jnp.where(active, nearest_neutral_d_raw / max_speed / time_norm, 0.0)

    roi = production / (ships + 1.0)
    roi_norm = roi / jnp.float32(2.0)
    is_high_value = (production >= 3.0).astype(jnp.float32) * active.astype(jnp.float32)

    ship_rank_all = _rank_norm(ships, active)
    prod_rank_all = _rank_norm(production, active)
    my_ship_rank = _rank_norm(ships, owner_is_me)
    enemy_ship_rank = _rank_norm(ships, owner_is_enemy)
    is_my_largest = owner_is_me & (my_ship_rank >= jnp.float32(1.0 - 1e-6))
    is_enemy_largest = owner_is_enemy & (enemy_ship_rank >= jnp.float32(1.0 - 1e-6))

    net_balance = ships + incoming_me - incoming_enemy
    net_balance_signed_log = jnp.sign(net_balance) * _ships_log_norm(jnp.abs(net_balance))
    would_lose = active & owner_is_me & (incoming_enemy > ships)

    comet_remaining = _comet_remaining_life(state)
    comet_remaining_norm = comet_remaining / jnp.float32(64.0)

    # Global variables computation
    my_planet_count = jnp.sum(owner_is_me.astype(jnp.float32))
    enemy_planet_count = jnp.sum(owner_is_enemy.astype(jnp.float32))
    neutral_planet_count = jnp.sum(owner_is_neutral.astype(jnp.float32))

    my_prod = jnp.sum(jnp.where(owner_is_me, production, 0.0))
    enemy_prod = jnp.sum(jnp.where(owner_is_enemy, production, 0.0))

    my_ships = jnp.sum(jnp.where(owner_is_me, ships, 0.0))
    enemy_ships = jnp.sum(jnp.where(owner_is_enemy, ships, 0.0))

    prod_lead = (my_prod - enemy_prod) / (my_prod + enemy_prod + 1.0)
    ship_lead = (my_ships - enemy_ships) / (my_ships + enemy_ships + 1.0)

    active_comets = jnp.sum(state.comets.active.astype(jnp.float32))

    fleet_owner = fleets[:, 1]
    fleet_ships = fleets[:, 6]
    fleet_active = fleets[:, 7] > 0.0
    f_is_me = fleet_active & (fleet_owner == player_f)
    f_is_enemy = fleet_active & (fleet_owner >= 0.0) & (fleet_owner != player_f)

    my_fleet_ships = jnp.sum(jnp.where(f_is_me, fleet_ships, 0.0))
    enemy_fleet_ships = jnp.sum(jnp.where(f_is_enemy, fleet_ships, 0.0))
    my_fleet_count = jnp.sum(f_is_me.astype(jnp.float32))
    enemy_fleet_count = jnp.sum(f_is_enemy.astype(jnp.float32))

    turn = state.step.astype(jnp.float32) / jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)

    my_largest_ships = jnp.max(jnp.where(owner_is_me, ships, 0.0))
    enemy_largest_ships = jnp.max(jnp.where(owner_is_enemy, ships, 0.0))
    my_largest_prod = jnp.max(jnp.where(owner_is_me, production, 0.0))
    enemy_largest_prod = jnp.max(jnp.where(owner_is_enemy, production, 0.0))

    cur_step = state.step.astype(jnp.float32)
    ep = jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)
    spawn_arr = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.float32)
    delta = spawn_arr - cur_step
    delta = jnp.where(delta > 0.0, delta, ep)
    next_comet_in = jnp.min(delta) / ep

    is_late_game = (turn > jnp.float32(0.5)).astype(jnp.float32)

    planet_features = jnp.stack(
        [
            active.astype(jnp.float32),                              # 0
            owner_is_me.astype(jnp.float32),                         # 1
            owner_is_enemy.astype(jnp.float32),                      # 2
            owner_is_neutral.astype(jnp.float32),                    # 3
            _ships_log_norm(ships),                                  # 4
            production / PRODUCTION_DENOM,                           # 5
            radius / RADIUS_DENOM,                                   # 6
            x / BOARD_SIZE,                                          # 7
            y / BOARD_SIZE,                                          # 8
            dx_c,                                                    # 9
            dy_c,                                                    # 10
            dist_c,                                                  # 11
            is_orbiting.astype(jnp.float32),                         # 12
            is_comet.astype(jnp.float32),                            # 13
            orbit_r_norm,                                            # 14
            orbit_angle_sin,                                         # 15
            orbit_angle_cos,                                         # 16
            _ships_log_norm(incoming_me),                            # 17
            _ships_log_norm(incoming_enemy),                         # 18
            eta_me_norm,                                             # 19
            eta_enemy_norm,                                          # 20
            roi_norm,                                                # 21
            is_high_value,                                           # 22
            nearest_enemy_d - nearest_friend_d,                      # 23
            ship_rank_all,                                           # 24
            prod_rank_all,                                           # 25
            net_balance_signed_log,                                  # 26
            would_lose.astype(jnp.float32),                          # 27
            time_to_nearest_enemy,                                   # 28
            time_to_nearest_neutral,                                 # 29
            is_my_largest.astype(jnp.float32),                       # 30
            is_enemy_largest.astype(jnp.float32),                    # 31
            comet_remaining_norm,                                    # 32
            jnp.broadcast_to(turn, (MAX_PLANETS,)),                                            # 33
            jnp.broadcast_to(1.0 - turn, (MAX_PLANETS,)),                                      # 34
            jnp.broadcast_to(my_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)),      # 35
            jnp.broadcast_to(enemy_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)),   # 36
            jnp.broadcast_to(neutral_planet_count / jnp.float32(MAX_PLANETS), (MAX_PLANETS,)), # 37
            jnp.broadcast_to(my_prod / jnp.float32(50.0), (MAX_PLANETS,)),                     # 38
            jnp.broadcast_to(enemy_prod / jnp.float32(50.0), (MAX_PLANETS,)),                  # 39
            jnp.broadcast_to(_ships_log_norm(my_ships), (MAX_PLANETS,)),                       # 40
            jnp.broadcast_to(_ships_log_norm(enemy_ships), (MAX_PLANETS,)),                    # 41
            jnp.broadcast_to(prod_lead, (MAX_PLANETS,)),                                       # 42
            jnp.broadcast_to(ship_lead, (MAX_PLANETS,)),                                       # 43
            jnp.broadcast_to(active_comets / jnp.float32(MAX_COMET_GROUPS), (MAX_PLANETS,)),   # 44
            jnp.broadcast_to(_fleet_ships_log_norm(my_fleet_ships), (MAX_PLANETS,)),           # 45
            jnp.broadcast_to(_fleet_ships_log_norm(enemy_fleet_ships), (MAX_PLANETS,)),        # 46
            jnp.broadcast_to(my_fleet_count / jnp.float32(MAX_FLEETS), (MAX_PLANETS,)),        # 47
            jnp.broadcast_to(enemy_fleet_count / jnp.float32(MAX_FLEETS), (MAX_PLANETS,)),     # 48
            jnp.broadcast_to(_ships_log_norm(my_largest_ships), (MAX_PLANETS,)),               # 49
            jnp.broadcast_to(_ships_log_norm(enemy_largest_ships), (MAX_PLANETS,)),            # 50
            jnp.broadcast_to(my_largest_prod / PRODUCTION_DENOM, (MAX_PLANETS,)),              # 51
            jnp.broadcast_to(enemy_largest_prod / PRODUCTION_DENOM, (MAX_PLANETS,)),           # 52
            jnp.broadcast_to(next_comet_in, (MAX_PLANETS,)),                                   # 53
            jnp.broadcast_to(is_late_game, (MAX_PLANETS,)),                                    # 54
        ],
        axis=-1,
    )
    planet_features = jnp.where(active[:, None], planet_features, 0.0)
    planet_mask = active

    return {
        "planet_features": planet_features,
        "planet_mask": planet_mask,
    }"""

match = re.search(r"def encode_observation\(.*?-> dict\[str, jnp\.ndarray\]:.*?    return \{.*?    \}", content, re.DOTALL)
if match:
    content = content[:match.start()] + new_encode + content[match.end():]
    with open('rl_training_jax/src/orbit_wars/features_jax.py', 'w') as f:
        f.write(content)
    print("Success")
else:
    print("Match failed")
