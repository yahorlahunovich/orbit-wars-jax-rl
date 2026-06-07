"""JAX simulation step for Orbit Wars (vectorized, vmap-friendly)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .comet import spawn_comet_for_state
from .constants import (
    CENTER,
    COMET_SPAWN_STEPS,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_COMET_PLANETS,
    MAX_FLEETS,
    MAX_MOVES_PER_PLAYER,
    MAX_PLANETS,
    NUM_PLAYERS,
    ROTATION_RADIUS_LIMIT,
)
from .convert import pack_comets
from .geometry import fleet_speed, in_bounds, sun_hit, swept_pair_hit
from .state import OrbitWarsState

# ---------------------------------------------------------------------------
# Python-side comet bookkeeping (rare: at most ~10 expiries + 5 spawns per game)
# ---------------------------------------------------------------------------


def remove_expired_comets(state: OrbitWarsState) -> OrbitWarsState:
    """Mark comet planets inactive when their path is exhausted.

    Pure-Python numpy implementation; cheap because it short-circuits when no
    comet groups are active.
    """
    active_groups = np.asarray(state.comets.active)
    if not active_groups.any():
        return state

    planets = np.asarray(state.planets).copy()
    initial = np.asarray(state.initial_planets).copy()
    planet_ids = np.asarray(state.comets.planet_ids)
    path_index = np.asarray(state.comets.path_index)
    path_lengths = np.asarray(state.comets.path_lengths)

    # Build a map planet_id -> slot for active planets once.
    active_mask = planets[:, 7] > 0.0
    id_to_slot = {int(planets[i, 0]): i for i in range(MAX_PLANETS) if active_mask[i]}

    mutated = False
    for gi in range(MAX_COMET_GROUPS):
        if not active_groups[gi]:
            continue
        idx_now = int(path_index[gi])
        for pi in range(4):
            pid = int(planet_ids[gi, pi])
            plen = int(path_lengths[gi, pi])
            if pid < 0 or plen <= 0 or idx_now < plen:
                continue
            slot = id_to_slot.get(pid)
            if slot is None:
                continue
            planets[slot, 7] = 0.0
            initial[slot, 7] = 0.0
            mutated = True

    if not mutated:
        return state
    return state.replace(planets=jnp.asarray(planets), initial_planets=jnp.asarray(initial))


def _maybe_spawn_comet_numpy(state: OrbitWarsState) -> OrbitWarsState:
    next_step = int(state.step) + 1
    if next_step not in COMET_SPAWN_STEPS or bool(state.done):
        return state

    spawn = spawn_comet_for_state(
        np.asarray(state.planets),
        int(state.n_planets),
        np.asarray(state.initial_planets),
        np.asarray(state.comet_planet_ids),
        int(state.n_comet_planet_ids),
        float(state.angular_velocity),
        next_step,
        int(state.episode_seed),
        comet_speed=4.0,
    )
    if spawn is None:
        return state

    planets = np.asarray(state.planets).copy()
    initial = np.asarray(state.initial_planets).copy()
    n = int(state.n_planets)
    for row in spawn["new_planets"]:
        if n >= MAX_PLANETS:
            break
        planets[n, :7] = row
        planets[n, 7] = 1.0
        initial[n, :7] = row
        initial[n, 7] = 1.0
        n += 1

    comet_ids = np.asarray(state.comet_planet_ids).copy()
    nc = int(state.n_comet_planet_ids)
    for pid in spawn["new_comet_ids"]:
        if nc >= MAX_COMET_PLANETS:
            break
        comet_ids[nc] = int(pid)
        nc += 1

    from .convert import _unpack_comets

    comets_list = _unpack_comets(state.comets)
    comets_list.append(spawn["group"])
    comets = pack_comets(comets_list)

    return state.replace(
        planets=jnp.asarray(planets),
        initial_planets=jnp.asarray(initial),
        n_planets=jnp.int32(n),
        comets=comets,
        comet_planet_ids=jnp.asarray(comet_ids),
        n_comet_planet_ids=jnp.int32(nc),
    )


# ---------------------------------------------------------------------------
# Vectorized JIT physics
# ---------------------------------------------------------------------------


def _apply_moves(
    state: OrbitWarsState,
    actions: jnp.ndarray,
    action_mask: jnp.ndarray,
    player: jnp.int32,
) -> OrbitWarsState:
    """Apply one player's moves sequentially."""
    planets = state.planets
    fleets = state.fleets
    n_fleets = state.n_fleets
    next_fleet_id = state.next_fleet_id

    pids_i32 = planets[:, 0].astype(jnp.int32)
    active_planet = planets[:, 7] > 0.0

    move_from = actions[:, 0].astype(jnp.int32)
    move_angle = actions[:, 1]
    move_ships_i32 = actions[:, 2].astype(jnp.int32)

    match = (move_from[:, None] == pids_i32[None, :]) & active_planet[None, :]
    source_idx = jnp.argmax(match.astype(jnp.int32), axis=-1)
    source_exists = jnp.any(match, axis=-1)

    use_move = (action_mask > 0.0) & (move_ships_i32 > 0) & source_exists

    player_f = player.astype(jnp.float32)

    def body(i, carry):
        planets_c, fleets_c, n_fleets_c, next_id_c = carry
        sidx = source_idx[i]
        safe_sidx = jnp.maximum(sidx, 0)
        owner = planets_c[safe_sidx, 1]
        have = planets_c[safe_sidx, 5].astype(jnp.int32)
        ships = move_ships_i32[i]
        valid = use_move[i] & (owner == player_f) & (have >= ships)

        new_ship_count = planets_c[safe_sidx, 5] - ships.astype(jnp.float32)
        planets_c = planets_c.at[safe_sidx, 5].set(
            jnp.where(valid, new_ship_count, planets_c[safe_sidx, 5])
        )
        radius = planets_c[safe_sidx, 4]
        start_x = planets_c[safe_sidx, 2] + jnp.cos(move_angle[i]) * (radius + 0.1)
        start_y = planets_c[safe_sidx, 3] + jnp.sin(move_angle[i]) * (radius + 0.1)

        slot = n_fleets_c
        can_add = valid & (slot < MAX_FLEETS)
        safe_slot = jnp.minimum(slot, MAX_FLEETS - 1)
        new_row = jnp.array(
            [
                next_id_c.astype(jnp.float32),
                player_f,
                start_x,
                start_y,
                move_angle[i],
                move_from[i].astype(jnp.float32),
                ships.astype(jnp.float32),
                1.0,
            ],
            dtype=jnp.float32,
        )
        fleets_c = fleets_c.at[safe_slot].set(
            jnp.where(can_add, new_row, fleets_c[safe_slot])
        )
        n_fleets_c = jnp.where(can_add, n_fleets_c + 1, n_fleets_c)
        next_id_c = jnp.where(can_add, next_id_c + 1, next_id_c)
        return (planets_c, fleets_c, n_fleets_c, next_id_c)

    planets, fleets, n_fleets, next_fleet_id = jax.lax.fori_loop(
        0, MAX_MOVES_PER_PLAYER, body, (planets, fleets, n_fleets, next_fleet_id),
    )
    return state.replace(
        planets=planets, fleets=fleets, n_fleets=n_fleets, next_fleet_id=next_fleet_id,
    )


def _production(state: OrbitWarsState) -> OrbitWarsState:
    owned = (state.planets[:, 1] >= 0.0) & (state.planets[:, 7] > 0.0)
    planets = state.planets.at[:, 5].set(
        jnp.where(owned, state.planets[:, 5] + state.planets[:, 6], state.planets[:, 5])
    )
    return state.replace(planets=planets)


def _compute_planet_paths(state: OrbitWarsState):
    """Returns (pids, old_x, old_y, new_x, new_y, radius, check)."""
    pids_i32 = state.planets[:, 0].astype(jnp.int32)
    old_x = state.planets[:, 2]
    old_y = state.planets[:, 3]
    radius = state.planets[:, 4]
    active = state.planets[:, 7] > 0.0

    cpids = state.comet_planet_ids
    valid_cpid = cpids >= 0
    is_comet = active & jnp.any(
        (pids_i32[:, None] == cpids[None, :]) & valid_cpid[None, :], axis=-1
    )

    init = state.initial_planets
    dx0 = init[:, 2] - CENTER
    dy0 = init[:, 3] - CENTER
    orbit_r = jnp.sqrt(dx0 * dx0 + dy0 * dy0)
    rotating = active & (orbit_r + radius < ROTATION_RADIUS_LIMIT) & (~is_comet)
    initial_angle = jnp.arctan2(dy0, dx0)
    current_angle = initial_angle + state.angular_velocity * state.step.astype(jnp.float32)
    rot_x = CENTER + orbit_r * jnp.cos(current_angle)
    rot_y = CENTER + orbit_r * jnp.sin(current_angle)
    new_x = jnp.where(rotating, rot_x, old_x)
    new_y = jnp.where(rotating, rot_y, old_y)
    check = jnp.where(active & (~is_comet), 1.0, 0.0)

    # ---- Comet path lookup ------------------------
    comets = state.comets
    cgpids = comets.planet_ids
    cplens = comets.path_lengths
    cactive = comets.active
    idx_g = comets.path_index + 1
    idx_clip = jnp.clip(idx_g, 0, MAX_COMET_PATH_LEN - 1)

    match_gqp = (
        (cgpids[..., None] == pids_i32[None, None, :])
        & active[None, None, :]
        & (cgpids[..., None] >= 0)
    )
    slot_gq = jnp.argmax(match_gqp.astype(jnp.int32), axis=-1)
    has_match_gq = jnp.any(match_gqp, axis=-1)

    on_board_gq = (idx_g[:, None] < cplens) & has_match_gq & cactive[:, None]
    is_comet_slot_gq = has_match_gq & cactive[:, None]

    g_arange = jnp.arange(MAX_COMET_GROUPS)
    q_arange = jnp.arange(4)
    g_grid, q_grid = jnp.meshgrid(g_arange, q_arange, indexing="ij")
    idx_grid = jnp.broadcast_to(idx_clip[:, None], (MAX_COMET_GROUPS, 4))
    path_xy = comets.paths[g_grid, q_grid, idx_grid, :]
    path_px = path_xy[..., 0]
    path_py = path_xy[..., 1]

    flat_slot = slot_gq.reshape(-1)
    flat_on = on_board_gq.reshape(-1)
    flat_px = path_px.reshape(-1)
    flat_py = path_py.reshape(-1)
    flat_is_comet_slot = is_comet_slot_gq.reshape(-1)

    new_x = new_x.at[flat_slot].set(jnp.where(flat_on, flat_px, new_x[flat_slot]))
    new_y = new_y.at[flat_slot].set(jnp.where(flat_on, flat_py, new_y[flat_slot]))

    # FIX: Newly placed comets (idx_g == 0) are not collision-eligible (Point 3).
    flat_idx_g = jnp.broadcast_to(idx_g[:, None], (MAX_COMET_GROUPS, 4)).reshape(-1)
    check_at_slot = jnp.where(flat_is_comet_slot & (flat_idx_g > 0), 1.0, check[flat_slot])
    check = check.at[flat_slot].set(check_at_slot)

    return pids_i32, old_x, old_y, new_x, new_y, radius, check


def _move_fleets(state, pids_i32, old_x, old_y, new_x, new_y, radius, check):
    """Returns (state, combat[planet, player])."""
    max_speed = state.ship_speed
    fleets = state.fleets

    active_f = fleets[:, 7] > 0.0
    owner = fleets[:, 1].astype(jnp.int32)
    angle = fleets[:, 4]
    ships = fleets[:, 6]
    old_fx = fleets[:, 2]
    old_fy = fleets[:, 3]
    speed = fleet_speed(ships, max_speed)
    new_fx = old_fx + jnp.cos(angle) * speed
    new_fy = old_fy + jnp.sin(angle) * speed

    hit_fp = swept_pair_hit(
        old_fx[:, None], old_fy[:, None], new_fx[:, None], new_fy[:, None],
        old_x[None, :], old_y[None, :], new_x[None, :], new_y[None, :], radius[None, :],
    )
    eligible = (check[None, :] > 0.0) & active_f[:, None]
    hit_fp = hit_fp & eligible

    any_hit = jnp.any(hit_fp, axis=-1)
    first_hit = jnp.argmax(hit_fp.astype(jnp.int32), axis=-1)
    hit_idx = jnp.where(any_hit, first_hit, -1)

    out_bounds = ~in_bounds(new_fx, new_fy)
    sun = sun_hit(old_fx, old_fy, new_fx, new_fy)
    remove = active_f & ((hit_idx >= 0) | out_bounds | sun)

    combat = jnp.zeros((MAX_PLANETS, NUM_PLAYERS), dtype=jnp.float32)
    hit_mask = active_f & (hit_idx >= 0)
    safe_hit_idx = jnp.where(hit_mask, hit_idx, 0)
    safe_owner = jnp.clip(owner, 0, NUM_PLAYERS - 1)
    contrib = jnp.where(hit_mask, ships, 0.0)
    combat = combat.at[safe_hit_idx, safe_owner].add(contrib)

    fleets = fleets.at[:, 2].set(new_fx)
    fleets = fleets.at[:, 3].set(new_fy)
    fleets = fleets.at[:, 7].set(jnp.where(remove, 0.0, fleets[:, 7]))
    return state.replace(fleets=fleets), combat


def _apply_planet_positions(state, new_x, new_y) -> OrbitWarsState:
    planets = state.planets.at[:, 2].set(new_x)
    planets = planets.at[:, 3].set(new_y)
    return state.replace(planets=planets)


def _advance_comet_indices(state: OrbitWarsState) -> OrbitWarsState:
    comets = state.comets
    comets = comets.replace(path_index=comets.path_index + 1)
    return state.replace(comets=comets)


def _expire_comets_in_jit(state: OrbitWarsState) -> OrbitWarsState:
    """Deactivate comet planets whose path has ended (vectorized, JIT-safe)."""
    comets = state.comets
    cgpids = comets.planet_ids
    cplens = comets.path_lengths
    cactive = comets.active
    idx_now = comets.path_index

    pids_i32 = state.planets[:, 0].astype(jnp.int32)
    active = state.planets[:, 7] > 0.0

    # FIX: Expire if the NEXT index would be out of bounds (Point 2)
    expired_gq = (idx_now[:, None] + 1 >= cplens) & (cplens > 0) & cactive[:, None] & (cgpids >= 0)
    match_gqp = (
        (cgpids[..., None] == pids_i32[None, None, :])
        & active[None, None, :]
        & (cgpids[..., None] >= 0)
    )
    slot_gq = jnp.argmax(match_gqp.astype(jnp.int32), axis=-1)
    has_match_gq = jnp.any(match_gqp, axis=-1)
    do_expire = (expired_gq & has_match_gq).reshape(-1)
    flat_slot = slot_gq.reshape(-1)

    planets = state.planets
    new_active = jnp.where(do_expire, 0.0, planets[flat_slot, 7])
    planets = planets.at[flat_slot, 7].set(new_active)
    initial = state.initial_planets
    new_active_i = jnp.where(do_expire, 0.0, initial[flat_slot, 7])
    initial = initial.at[flat_slot, 7].set(new_active_i)
    return state.replace(planets=planets, initial_planets=initial)


def _resolve_combat(state: OrbitWarsState, combat: jnp.ndarray) -> OrbitWarsState:
    planets = state.planets
    active = planets[:, 7] > 0.0
    ships_p0 = combat[:, 0]
    ships_p1 = combat[:, 1]
    total = ships_p0 + ships_p1
    contested = active & (total > 0.0)

    top = jnp.maximum(ships_p0, ships_p1)
    second = jnp.minimum(ships_p0, ships_p1)
    tie = ships_p0 == ships_p1
    survivor_ships = jnp.where(tie, 0.0, top - second)
    top_player = jnp.where(ships_p0 >= ships_p1, 0, 1).astype(jnp.int32)
    survivor_owner = jnp.where(survivor_ships > 0.0, top_player, -1)

    owner = planets[:, 1].astype(jnp.int32)
    same = owner == survivor_owner
    diff = (~same) & (survivor_owner >= 0)

    new_ships_same = planets[:, 5] + survivor_ships
    new_ships_diff = planets[:, 5] - survivor_ships
    captured = new_ships_diff < 0.0
    new_ships = jnp.where(
        same,
        new_ships_same,
        jnp.where(captured, jnp.abs(new_ships_diff), new_ships_diff),
    )
    new_owner = jnp.where(
        contested & diff & captured,
        survivor_owner.astype(jnp.float32),
        planets[:, 1],
    )
    planets = planets.at[:, 5].set(jnp.where(contested, new_ships, planets[:, 5]))
    planets = planets.at[:, 1].set(jnp.where(contested & diff, new_owner, planets[:, 1]))
    return state.replace(planets=planets)


def _termination(state: OrbitWarsState) -> OrbitWarsState:
    step = state.step
    terminated_by_steps = step >= (state.episode_steps - 1)

    owned = (state.planets[:, 1] >= 0.0) & (state.planets[:, 7] > 0.0)
    fleet_active = state.fleets[:, 7] > 0.0
    fleet_owners = state.fleets[:, 1].astype(jnp.int32)
    planet_owners = state.planets[:, 1].astype(jnp.int32)

    p_range = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
    owned_match = owned[None, :] & (planet_owners[None, :] == p_range[:, None])
    fleet_match = fleet_active[None, :] & (fleet_owners[None, :] == p_range[:, None])

    has_any = jnp.any(owned_match, axis=-1) | jnp.any(fleet_match, axis=-1)
    alive_count = jnp.sum(has_any.astype(jnp.int32))
    terminated = terminated_by_steps | (alive_count <= 1)

    planet_score = jnp.sum(jnp.where(owned_match, state.planets[None, :, 5], 0.0), axis=-1)
    fleet_score = jnp.sum(jnp.where(fleet_match, state.fleets[None, :, 6], 0.0), axis=-1)
    scores = planet_score + fleet_score

    max_score = jnp.max(scores)
    all_max = jnp.all(scores == max_score)
    
    rewards = jnp.where(
        terminated,
        jnp.where(
            all_max | (max_score <= 0.0),
            jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
            jnp.where(scores == max_score, 1.0, -1.0)
        ),
        jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32),
    )
    return state.replace(done=terminated, rewards=rewards)


# ---------------------------------------------------------------------------
# Public step API
# ---------------------------------------------------------------------------


@jax.jit
def step_jit(
    state: OrbitWarsState,
    actions_p0: jnp.ndarray,
    actions_p1: jnp.ndarray,
    mask_p0: jnp.ndarray,
    mask_p1: jnp.ndarray,
) -> OrbitWarsState:
    # 1. Expire comets before processing moves (Point 2)
    state = _expire_comets_in_jit(state)
    
    state = _apply_moves(state, actions_p0, mask_p0, jnp.int32(0))
    state = _apply_moves(state, actions_p1, mask_p1, jnp.int32(1))
    state = _production(state)
    pids, old_x, old_y, new_x, new_y, radius, check = _compute_planet_paths(state)
    state, combat = _move_fleets(state, pids, old_x, old_y, new_x, new_y, radius, check)
    state = _apply_planet_positions(state, new_x, new_y)
    
    # 2. Advance indices for the NEXT step
    state = _advance_comet_indices(state)
    
    state = _resolve_combat(state, combat)
    state = state.replace(step=state.step + 1)
    state = _termination(state)
    return state


def step(
    state: OrbitWarsState,
    actions: list[list[list[float | int]]] | None = None,
    *,
    actions_p0: jnp.ndarray | None = None,
    actions_p1: jnp.ndarray | None = None,
    mask_p0: jnp.ndarray | None = None,
    mask_p1: jnp.ndarray | None = None,
) -> OrbitWarsState:
    state = _maybe_spawn_comet_numpy(state)

    if actions is not None:
        a0, m0 = _list_action_to_padded(actions[0])
        a1, m1 = _list_action_to_padded(actions[1])
    else:
        a0, m0 = actions_p0, mask_p0
        a1, m1 = actions_p1, mask_p1

    state = step_jit(state, a0, a1, m0, m1)
    return state


def _list_action_to_padded(moves: list[list[float | int]]) -> tuple[jnp.ndarray, jnp.ndarray]:
    arr = np.zeros((MAX_MOVES_PER_PLAYER, 3), dtype=np.float32)
    mask = np.zeros((MAX_MOVES_PER_PLAYER,), dtype=np.float32)
    n = min(len(moves), MAX_MOVES_PER_PLAYER)
    for i in range(n):
        move = moves[i]
        if len(move) != 3:
            continue
        arr[i, 0] = float(move[0])
        arr[i, 1] = float(move[1])
        arr[i, 2] = float(move[2])
        mask[i] = 1.0
    return jnp.asarray(arr), jnp.asarray(mask)


@jax.jit
def batched_step(
    states: OrbitWarsState,
    actions_p0: jnp.ndarray,
    actions_p1: jnp.ndarray,
    mask_p0: jnp.ndarray,
    mask_p1: jnp.ndarray,
) -> OrbitWarsState:
    return jax.vmap(step_jit)(states, actions_p0, actions_p1, mask_p0, mask_p1)
