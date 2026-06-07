import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32, Bool
from env.jax_orbit_wars import JaxEnvState, plan_target_shot_jax, fleet_speed_jax, CENTER, SUN_RADIUS

MAX_SHIPS = 400.0
MAX_PRODUCTION = 5.0
MIN_LAUNCH_SHIPS = 5.0
FUTURE_ORACLE_STEPS = 32
FUTURE_ORACLE_SCALE = 30.0
N_SHIP_OPTIONS = 3


@jax.jit
def ship_options_for_edge_jax(src_ships: Array, tgt_ships: Array, tgt_owner: Array, player_id: Array) -> Array:
    """Returns the 3 source-fraction bins: 50%, 75%, and 100% of source ships."""
    pct_50 = jnp.maximum(1.0, jnp.round(0.50 * src_ships))
    pct_75 = jnp.maximum(1.0, jnp.round(0.75 * src_ships))
    pct_100 = src_ships

    return jnp.stack([pct_50, pct_75, pct_100], axis=-1)


@jax.jit
def extract_node_features_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Extracts 12 player-index-invariant node features for all 60 planets."""
    cur_turn = state.cur_turn
    owners = state.future_timeline[:, cur_turn, 0]
    ships = state.future_timeline[:, cur_turn, 1]
    positions = state.planet_positions_all_turns[:, cur_turn]
    
    is_owned_by_player = (owners == player_id).astype(jnp.float32)
    is_owned_by_opponent = ((owners != player_id) & (owners != -1)).astype(jnp.float32)
    is_neutral = (owners == -1).astype(jnp.float32)
    
    x_norm = positions[:, 0] / 100.0
    y_norm = positions[:, 1] / 100.0
    r_norm = jnp.minimum(state.planet_radius / 10.0, 1.0)
    s_norm = jnp.minimum(ships / MAX_SHIPS, 1.0)
    p_norm = jnp.minimum(state.planet_production / MAX_PRODUCTION, 1.0)
    
    is_rot = state.is_rotating.astype(jnp.float32)
    is_com = state.is_comet.astype(jnp.float32)
    
    dist_to_sun = jnp.sqrt(jnp.sum((positions - CENTER) ** 2, axis=1)) / 70.71
    
    # Dynamic Fleet Pressure from precomputed future incoming fleets
    inc = state.incoming_fleets
    
    # Mask out past turns
    turns_mask = (jnp.arange(500) > cur_turn)[None, :, None]
    valid_inc = jnp.where(turns_mask, inc, 0.0)
    
    # Sum over all opponents
    opp_arrivals = jnp.sum(valid_inc, axis=-1) - valid_inc[:, :, player_id]
    net_incoming = jnp.sum(opp_arrivals - valid_inc[:, :, player_id], axis=1)
    fleet_pressure = jnp.clip(net_incoming / MAX_SHIPS, -1.0, 1.0)
    
    feats = jnp.stack([
        is_owned_by_player,
        is_owned_by_opponent,
        is_neutral,
        x_norm,
        y_norm,
        r_norm,
        s_norm,
        p_norm,
        is_rot,
        is_com,
        dist_to_sun,
        fleet_pressure
    ], axis=-1)
    
    # Apply active mask to zero out inactive slots, except neutral/dist_to_sun
    feats = feats * state.active_mask[:, None]
    feats = feats.at[:, 2].set(jnp.where(state.active_mask, feats[:, 2], 1.0))
    feats = feats.at[:, 10].set(jnp.where(state.active_mask, feats[:, 10], 1.0))
    
    return feats


@jax.jit
def extract_node_features_v8_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Extracts V8's 21 node features: 12 base features plus 3 facts for each ship bucket."""
    base = extract_node_features_jax(state, player_id)
    cur_turn = state.cur_turn
    ships = state.future_timeline[:, cur_turn, 1]

    def bucket_block(pct):
        bucket_ships = jnp.maximum(1.0, jnp.floor(pct * ships + 0.5))
        speed = fleet_speed_jax(bucket_ships)
        return jnp.stack(
            [
                jnp.minimum(bucket_ships / MAX_SHIPS, 1.0),
                speed / 6.0,
                (bucket_ships >= 20.0).astype(jnp.float32),
            ],
            axis=-1,
        )

    buckets = jnp.concatenate(
        [
            bucket_block(0.50),
            bucket_block(0.75),
            bucket_block(1.00),
        ],
        axis=-1,
    )
    buckets = buckets * state.active_mask[:, None]
    return jnp.concatenate([base, buckets], axis=-1)


@jax.jit
def extract_future_sight_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Extracts sliding-window 32-step future timelines in JAX."""
    cur_turn = state.cur_turn
    timeline = state.future_timeline  # (60, 500, 2)
    
    # 32-step slicing
    t_indices = jnp.clip(cur_turn + 1 + jnp.arange(FUTURE_ORACLE_STEPS), 0, 499)
    
    # Sliced owners and ships: (60, 32)
    sliced_owners = timeline[:, t_indices, 0]
    sliced_ships = timeline[:, t_indices, 1]
    
    is_player = sliced_owners == player_id
    is_opponent = (sliced_owners != player_id) & (sliced_owners != -1)
    
    val = jnp.where(is_player, sliced_ships, jnp.where(is_opponent, -sliced_ships, 0.0))
    future_sight = val / FUTURE_ORACLE_SCALE
    future_sight = future_sight * state.active_mask[:, None]
    
    return future_sight


@jax.jit
def extract_edge_features_jax(state: JaxEnvState, player_id: Array) -> tuple[Array, Array, Array]:
    """Generates all-to-all edge index and 9 edge features in JAX."""
    cur_turn = state.cur_turn
    positions = state.planet_positions_all_turns[:, cur_turn]
    ships = state.future_timeline[:, cur_turn, 1]
    owners = state.future_timeline[:, cur_turn, 0]
    
    # Create all pair combinations
    p_count = 60
    src, dst = jnp.meshgrid(jnp.arange(p_count), jnp.arange(p_count), indexing="ij")
    src = src.reshape(-1)
    dst = dst.reshape(-1)
    
    # Edge index: (2, 3600)
    edge_index = jnp.stack([src, dst], axis=0)
    
    diff = positions[dst] - positions[src]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    direct_angle = jnp.arctan2(diff[:, 1], diff[:, 0])
    
    # Ship options and speeds
    src_ships = ships[src]
    tgt_ships = ships[dst]
    tgt_owners = owners[dst]
    
    options = ship_options_for_edge_jax(src_ships, tgt_ships, tgt_owners, player_id)
    ref_ships = options[:, 0]  # 50% source bucket as static reference ship count
    
    speed = fleet_speed_jax(ref_ships)
    
    sin_a = jnp.sin(direct_angle)
    cos_a = jnp.cos(direct_angle)
    launch_clearance = state.planet_radius[src] + 0.1
    start_x = positions[src, 0] + cos_a * launch_clearance
    start_y = positions[src, 1] + sin_a * launch_clearance
    
    to_target_x = positions[dst, 0] - start_x
    to_target_y = positions[dst, 1] - start_y
    launch_dist = jnp.sqrt(to_target_x * to_target_x + to_target_y * to_target_y)
    turns = launch_dist / jnp.maximum(speed, 1e-6)
    
    # Check sun intersection
    seg_x = positions[dst, 0] - start_x
    seg_y = positions[dst, 1] - start_y
    seg_len_sq = jnp.maximum(seg_x * seg_x + seg_y * seg_y, 1e-9)
    proj = ((CENTER - start_x) * seg_x + (CENTER - start_y) * seg_y) / seg_len_sq
    proj = jnp.clip(proj, 0.0, 1.0)
    close_x = start_x + proj * seg_x
    close_y = start_y + proj * seg_y
    sun_dist = jnp.sqrt((CENTER - close_x) ** 2 + (CENTER - close_y) ** 2)
    crosses_sun = (sun_dist < SUN_RADIUS).astype(jnp.float32)
    
    # Construct Edge Features: (3600, 9)
    feat0 = diff[:, 0] / 100.0
    feat1 = diff[:, 1] / 100.0
    feat2 = dist / (100.0 * jnp.sqrt(2.0))
    feat3 = sin_a
    feat4 = cos_a
    feat5 = jnp.minimum(ref_ships / MAX_SHIPS, 1.0)
    
    desired_ships = jnp.maximum(tgt_ships + 1.0, 20.0)
    feat6 = (src_ships >= desired_ships).astype(jnp.float32)
    feat7 = crosses_sun
    feat8 = jnp.minimum(turns / 100.0, 1.0)
    
    edge_features = jnp.stack([
        feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8
    ], axis=-1)
    
    return edge_index, edge_features, options


@jax.jit
def extract_edge_features_v8_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Generate V8's 14 all-to-all edge features."""
    cur_turn = state.cur_turn
    _edge_index, base_flat, _ = extract_edge_features_jax(state, player_id)
    base = base_flat.reshape(60, 60, 9)

    positions = state.planet_positions_all_turns[:, cur_turn]
    ships = state.future_timeline[:, cur_turn, 1]
    diff = positions[None, :, :] - positions[:, None, :]
    direct = jnp.arctan2(diff[..., 1], diff[..., 0])
    sin_a = jnp.sin(direct)
    cos_a = jnp.cos(direct)
    start_x = positions[:, None, 0] + cos_a * (state.planet_radius[:, None] + 0.1)
    start_y = positions[:, None, 1] + sin_a * (state.planet_radius[:, None] + 0.1)
    seg_x = positions[None, :, 0] - start_x
    seg_y = positions[None, :, 1] - start_y
    rough_dist = jnp.sqrt(seg_x * seg_x + seg_y * seg_y + 1e-9)

    src_ships = ships[:, None]
    tgt_ships = ships[None, :]

    def bucket_edge_block(pct):
        bucket_ships = jnp.maximum(1.0, jnp.floor(pct * src_ships + 0.5))
        speed = fleet_speed_jax(bucket_ships)
        rough_turns = jnp.minimum(rough_dist / jnp.maximum(speed, 1e-6) / 100.0, 1.0)
        can_clear = (bucket_ships >= (tgt_ships + 1.0)).astype(jnp.float32)
        return rough_turns, can_clear

    turns50, clear50 = bucket_edge_block(0.50)
    turns75, clear75 = bucket_edge_block(0.75)
    turns100, clear100 = bucket_edge_block(1.00)
    ratio = jnp.clip(src_ships / jnp.maximum(tgt_ships, 1.0), 0.0, 20.0) / 20.0
    roi = jnp.zeros((60, 60), dtype=jnp.float32)

    geometry = base[..., jnp.array([0, 1, 2, 3, 4, 7])]
    edges = jnp.concatenate(
        [
            geometry,
            roi[..., None],
            ratio[..., None],
            turns50[..., None],
            turns75[..., None],
            turns100[..., None],
            clear50[..., None],
            clear75[..., None],
            clear100[..., None],
        ],
        axis=-1,
    )
    edge_mask = (state.active_mask[:, None] & state.active_mask[None, :])[:, :, None]
    return edges * edge_mask


@jax.jit
def extract_owned_nodes_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Finds up to 60 active planet indices owned by the player, padded with -1."""
    cur_turn = state.cur_turn
    owners = state.future_timeline[:, cur_turn, 0]
    is_active = state.active_mask
    
    is_owned = (owners == player_id) & is_active
    owned_indices = jnp.where(is_owned, jnp.arange(60), 999)
    
    # Sort to bring owned nodes to the front
    sorted_owned = jnp.sort(owned_indices)
    
    # Replace padding with -1
    final_owned = jnp.where(sorted_owned < 60, sorted_owned, -1)

    return final_owned[:60]


@jax.jit
def compute_edge_valid_mask_jax(
    state: JaxEnvState,
    owned_nodes: Array,
    player_id: Array,
) -> Array:
    """Construct the (60, 60, N_SHIP_OPTIONS) action mask from trajectory legality."""
    cur_turn = state.cur_turn
    positions = state.planet_positions_all_turns[:, cur_turn]  # (60, 2)
    ids = jnp.arange(60)
    sun_center = jnp.array([CENTER, CENTER])

    def check_slot(slot):
        src = owned_nodes[slot]
        src_safe = jnp.where(src >= 0, src, 0)
        
        src_ships = state.future_timeline[src_safe, cur_turn, 1]
        src_valid = src >= 0
        src_pos = positions[src_safe]
        seg = positions - src_pos[None, :]  # (60, 2)
        seg_len_sq = jnp.maximum(jnp.sum(seg * seg, axis=-1), 1e-9)  # (60,)

        to_sun = sun_center - src_pos
        sun_proj = jnp.clip(jnp.sum(to_sun[None, :] * seg, axis=-1) / seg_len_sq, 0.0, 1.0)
        sun_closest = src_pos[None, :] + sun_proj[:, None] * seg
        sun_dist = jnp.sqrt(jnp.sum((sun_center[None, :] - sun_closest) ** 2, axis=-1))
        blocks_sun = sun_dist < SUN_RADIUS

        blocker_vec = positions[None, :, :] - src_pos[None, None, :]  # (1, 60, 2)
        seg_t = seg[:, None, :]  # (60, 1, 2)
        proj = jnp.sum(blocker_vec * seg_t, axis=-1) / seg_len_sq[:, None]  # (60, 60)
        closest = src_pos[None, None, :] + proj[:, :, None] * seg_t
        blocker_dist = jnp.sqrt(jnp.sum((positions[None, :, :] - closest) ** 2, axis=-1))
        not_endpoint = (ids[None, :] != src_safe) & (ids[None, :] != ids[:, None])
        blocks_planet = (
            state.active_mask[None, :]
            & not_endpoint
            & (proj > 0.0)
            & (proj < 1.0)
            & (blocker_dist <= state.planet_radius[None, :])
        )
        path_blocked = blocks_sun | jnp.any(blocks_planet, axis=-1)
        
        def check_target(tgt):
            tgt_active = state.active_mask[tgt]
            is_self = src == tgt
            not_self = ~is_self
            
            base_valid = src_valid & tgt_active & not_self & (~path_blocked[tgt])
            
            opts = ship_options_for_edge_jax(src_ships, tgt_ships=state.future_timeline[tgt, cur_turn, 1], tgt_owner=state.future_timeline[tgt, cur_turn, 0], player_id=player_id)
            
            def check_opt(opt):
                ships = opts[opt]
                ships_valid = (ships >= MIN_LAUNCH_SHIPS) & (ships <= src_ships)
                noop_valid = src_valid & is_self & (opt == 0)
                return noop_valid | (base_valid & ships_valid)
            
            return jax.vmap(check_opt)(jnp.arange(N_SHIP_OPTIONS))
        
        return jax.vmap(check_target)(jnp.arange(60))
    
    mask = jax.vmap(check_slot)(jnp.arange(60))  # (60, 60, N_SHIP_OPTIONS)
    return mask


@jax.jit
def compute_edge_valid_mask_raytrace_jax(
    state: JaxEnvState,
    owned_nodes: Array,
    player_id: Array,
) -> Array:
    """Constructs the (60, 60, N_SHIP_OPTIONS) exact target-shot validity mask."""
    cur_turn = state.cur_turn
    
    def check_validity(slot, tgt, opt):
        src = owned_nodes[slot]
        src_valid = src >= 0
        is_self = src == tgt
        tgt_valid = state.active_mask[tgt] & (~is_self)
        base_valid = src_valid & tgt_valid
        
        src_ships = state.future_timeline[src, cur_turn, 1]
        tgt_ships = state.future_timeline[tgt, cur_turn, 1]
        tgt_owner = state.future_timeline[tgt, cur_turn, 0]
        
        opts = ship_options_for_edge_jax(src_ships, tgt_ships, tgt_owner, player_id)
        ships = opts[opt]
        ships_valid = (ships >= MIN_LAUNCH_SHIPS) & (ships <= src_ships)
        noop_valid = src_valid & is_self & (opt == 0)
        valid = noop_valid | (base_valid & ships_valid)
        
        _, _, _, planned_viable = plan_target_shot_jax(
            state,
            src,
            tgt,
            ships,
            cur_turn,
        )
        
        return noop_valid | (valid & planned_viable)

    mask = jax.vmap(
        lambda slot: jax.vmap(
            lambda tgt: jax.vmap(
                lambda opt: check_validity(slot, tgt, opt)
            )(jnp.arange(N_SHIP_OPTIONS))
        )(jnp.arange(60))
    )(jnp.arange(60))
    
    return mask


@jax.jit
def extract_global_features_jax(state: JaxEnvState, player_id: Array) -> Array:
    """Extracts 8 global game state features for the player."""
    cur_turn = state.cur_turn
    
    owners = state.future_timeline[:, cur_turn, 0]
    ships = state.future_timeline[:, cur_turn, 1]
    active = state.active_mask
    
    my_planets = (owners == player_id) & active
    enemy_planets = (owners != player_id) & (owners != -1) & active
    neutral_planets = (owners == -1) & active
    
    MAX_P = 60.0
    MAX_S = 400.0
    denom = MAX_P * MAX_S
    
    # Approximate fleet totals from incoming_fleets (future arrivals)
    turns_mask = (jnp.arange(500) > cur_turn)[None, :, None]
    valid_inc = jnp.where(turns_mask, state.incoming_fleets, 0.0)
    my_fleet_ships = jnp.sum(valid_inc[:, :, player_id])
    enemy_fleet_ships = jnp.sum(valid_inc) - my_fleet_ships
    
    return jnp.stack([
        jnp.minimum(cur_turn / 500.0, 1.0),
        my_planets.sum() / MAX_P,
        enemy_planets.sum() / MAX_P,
        neutral_planets.sum() / MAX_P,
        jnp.sum(jnp.where(my_planets, ships, 0.0)) / denom,
        jnp.sum(jnp.where(enemy_planets, ships, 0.0)) / denom,
        jnp.minimum(my_fleet_ships / denom, 1.0),
        jnp.minimum(enemy_fleet_ships / denom, 1.0),
    ])


@jax.jit
def state_to_graph_jax(state: JaxEnvState, player_id: Array) -> tuple[
    Array,  # node_features (60, 12)
    Array,  # edge_index (2, 3600)
    Array,  # edge_features (3600, 9)
    Array,  # future_sight (60, 32)
    Array,  # global_features (8,)
    Array,  # owned_nodes (60,)
    Array,  # edge_valid_mask (60, 60, N_SHIP_OPTIONS) - simplified
]:
    """Top-level JAX-compiled features extractor."""
    node_features = extract_node_features_jax(state, player_id)
    edge_index, edge_features, _ = extract_edge_features_jax(state, player_id)
    future_sight = extract_future_sight_jax(state, player_id)
    global_features = extract_global_features_jax(state, player_id)
    owned_nodes = extract_owned_nodes_jax(state, player_id)
    edge_valid_mask = compute_edge_valid_mask_jax(state, owned_nodes, player_id)
    
    return node_features, edge_index, edge_features, future_sight, global_features, owned_nodes, edge_valid_mask


from typing import NamedTuple

class ObsBatch(NamedTuple):
    node_features: jnp.ndarray    # (60, 21)
    edge_features: jnp.ndarray    # (60, 60, 14)
    future_sight: jnp.ndarray     # (60, 32)
    global_features: jnp.ndarray  # (8,)
    owned_nodes: jnp.ndarray      # (60,) padded with -1
    edge_valid_mask: jnp.ndarray  # (60, 60, 3)


@jax.jit
def extract_obs_v8_jax(state: JaxEnvState, player_id: Array) -> ObsBatch:
    node_features = extract_node_features_v8_jax(state, player_id)
    edge_features = extract_edge_features_v8_jax(state, player_id)
    future_sight = extract_future_sight_jax(state, player_id)
    global_features = extract_global_features_jax(state, player_id)
    owned_nodes = extract_owned_nodes_jax(state, player_id)
    edge_valid_mask = compute_edge_valid_mask_jax(state, owned_nodes, player_id)
    
    return ObsBatch(
        node_features=node_features,
        edge_features=edge_features,
        future_sight=future_sight,
        global_features=global_features,
        owned_nodes=owned_nodes,
        edge_valid_mask=edge_valid_mask,
    )


@jax.jit
def extract_obs_v9_jax(state: JaxEnvState, player_id: Array) -> ObsBatch:
    obs = extract_obs_v8_jax(state, player_id)
    return ObsBatch(
        node_features=obs.node_features,
        edge_features=obs.edge_features.astype(jnp.bfloat16),
        future_sight=obs.future_sight.astype(jnp.bfloat16),
        global_features=obs.global_features,
        owned_nodes=obs.owned_nodes,
        edge_valid_mask=obs.edge_valid_mask,
    )

