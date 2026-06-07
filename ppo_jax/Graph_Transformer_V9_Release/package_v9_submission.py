"""Build a single-file Kaggle submission for GraphTransformerV9.

The legacy ``package_jax_submission.py`` targets planet_transformer_jax/V6
NumPy weights. V9 changed the feature contract and policy heads, so this
script generates a dedicated NumPy-only V9 ``main.py`` with embedded NPZ
weights.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


V9_INIT = r'''
import base64
import io

_MODEL_B64 = """__CKPT_B64__"""
W = np.load(io.BytesIO(base64.b64decode(_MODEL_B64.encode("ascii"))))
_DECODE_MODE = "__DECODE_MODE__"
_LAUNCH_TEMP = __LAUNCH_TEMP__  # temperature for the launch/hold decision
_RNG = np.random.default_rng(int(os.environ.get("GRAPH_TRANSFORMER_V9_SEED", "20260528")))
HIDDEN_DIM = int(W["node_ln_w"].shape[0])
N_SHIP_OPTIONS = 3
MODEL_INPUT_DIM = int(W["node_encoder.w_proj"].shape[1])
NODE_FEATURE_DIM = MODEL_INPUT_DIM - 32 - 8
EDGE_FEATURE_DIM = int(W["edge_encoder.w1"].shape[1])
FUTURE_ORACLE_STEPS = 32
assert NODE_FEATURE_DIM == 21, f"[graph_transformer_v9] expected 21 node features, got {NODE_FEATURE_DIM}"
assert EDGE_FEATURE_DIM == 14, f"[graph_transformer_v9] expected 14 edge features, got {EDGE_FEATURE_DIM}"
EDGE_DIM = int(W["edge_ln_w"].shape[0])
HEADS = 4
HEAD_DIM = HIDDEN_DIM // HEADS
N_LAYERS = 0
while f"layers.{N_LAYERS}.w_q" in W:
    N_LAYERS += 1
'''


V9_FORWARD = r'''
def _forward(node_features, edge_features, future_sight, global_features, owned_nodes):
    gf = np.broadcast_to(global_features[None, :], (MAX_PLANETS, 8))
    h = np.concatenate([node_features, future_sight, gf], axis=-1).astype(np.float32)
    h = _ln(h)
    h = _silu(h @ W["node_encoder.w_proj"].T + W["node_encoder.b_proj"])
    sw = h @ W["node_encoder.swiglu.w_in"].T + W["node_encoder.swiglu.b_in"]
    val, gate = np.split(sw, 2, axis=-1)
    h = h + (val * _silu(gate)) @ W["node_encoder.swiglu.w_out"].T + W["node_encoder.swiglu.b_out"]
    node_h = _ln(h @ W["node_encoder.w_out"].T + W["node_encoder.b_out"])

    for i in range(N_LAYERS):
        p = f"layers.{i}"
        q_norm = _ln(node_h, W[f"{p}.ln_w"], W[f"{p}.ln_b"])
        q = (q_norm @ W[f"{p}.w_q"].T + W[f"{p}.b_q"]).reshape(MAX_PLANETS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = (q_norm @ W[f"{p}.w_k"].T + W[f"{p}.b_k"]).reshape(MAX_PLANETS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = (q_norm @ W[f"{p}.w_v"].T + W[f"{p}.b_v"]).reshape(MAX_PLANETS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * (HEAD_DIM ** -0.5)
        attn = _softmax(scores, axis=-1)
        context = np.matmul(attn, v).transpose(1, 0, 2).reshape(MAX_PLANETS, HIDDEN_DIM)
        node_h = node_h + (context @ W[f"{p}.w_out"].T + W[f"{p}.b_out"])
        h_norm = _ln(node_h, W[f"{p}.ffn_ln_w"], W[f"{p}.ffn_ln_b"])
        node_h = node_h + (_silu(h_norm @ W[f"{p}.w_ffn1"].T + W[f"{p}.b_ffn1"]) @ W[f"{p}.w_ffn2"].T + W[f"{p}.b_ffn2"])

    node_h = _ln(node_h, W["node_ln_w"], W["node_ln_b"])

    flat_e = edge_features.reshape(-1, EDGE_FEATURE_DIM).astype(np.float32)
    edge_h = _silu(_ln(flat_e) @ W["edge_encoder.w1"].T + W["edge_encoder.b1"])
    edge_h = _ln(edge_h) @ W["edge_encoder.w2"].T + W["edge_encoder.b2"]
    edge_h = edge_h.reshape(MAX_PLANETS, MAX_PLANETS, EDGE_DIM)
    edge_h = _ln(edge_h.reshape(-1, EDGE_DIM), W["edge_ln_w"], W["edge_ln_b"]).reshape(MAX_PLANETS, MAX_PLANETS, EDGE_DIM)

    head = "policy_heads.0"
    src_safe = np.where(owned_nodes >= 0, owned_nodes, 0)
    src_h = node_h[src_safe]
    edge_s = edge_h[src_safe]
    raw_edge_s = edge_features[src_safe]

    send_cat = np.concatenate([
        src_h,
        np.broadcast_to(global_features[None, :], (MAX_OWNED, 8)),
    ], axis=-1)
    send = (_silu(send_cat @ W[f"{head}.w_send1"].T + W[f"{head}.b_send1"]) @ W[f"{head}.w_send2"].T + W[f"{head}.b_send2"]).reshape(MAX_OWNED)

    pair_cat = np.concatenate([
        np.broadcast_to(src_h[:, None, :], (MAX_OWNED, MAX_PLANETS, HIDDEN_DIM)),
        np.broadcast_to(node_h[None, :, :], (MAX_OWNED, MAX_PLANETS, HIDDEN_DIM)),
        edge_s,
        np.broadcast_to(global_features[None, None, :], (MAX_OWNED, MAX_PLANETS, 8)),
    ], axis=-1)
    flat_pair = pair_cat.reshape(-1, 2 * HIDDEN_DIM + EDGE_DIM + 8)
    target_base = (
        _silu(flat_pair @ W[f"{head}.w_tgt1"].T + W[f"{head}.b_tgt1"]) @ W[f"{head}.w_tgt2"].T + W[f"{head}.b_tgt2"]
    ).reshape(MAX_OWNED, MAX_PLANETS)

    roi = raw_edge_s[:, :, 6:7]
    ratio = raw_edge_s[:, :, 7:8]
    turns = raw_edge_s[:, :, 8:11]
    clear = raw_edge_s[:, :, 11:14]
    bucket_raw = np.stack(
        [
            np.concatenate([roi, ratio, turns[:, :, i:i + 1], clear[:, :, i:i + 1]], axis=-1)
            for i in range(N_SHIP_OPTIONS)
        ],
        axis=2,
    ).reshape(MAX_OWNED, MAX_PLANETS, N_SHIP_OPTIONS, 4)

    bucket_cat = np.concatenate(
        [
            np.broadcast_to(pair_cat[:, :, None, :], (MAX_OWNED, MAX_PLANETS, N_SHIP_OPTIONS, 2 * HIDDEN_DIM + EDGE_DIM + 8)),
            bucket_raw,
        ],
        axis=-1,
    )
    flat_bucket = bucket_cat.reshape(-1, 2 * HIDDEN_DIM + EDGE_DIM + 8 + 4)
    bucket_utility = (
        _silu(flat_bucket @ W[f"{head}.w_bucket1"].T + W[f"{head}.b_bucket1"]) @ W[f"{head}.w_bucket2"].T + W[f"{head}.b_bucket2"]
    ).reshape(MAX_OWNED, MAX_PLANETS, N_SHIP_OPTIONS)
    target = target_base + np.max(bucket_utility, axis=-1)
    frac = (
        _silu(flat_bucket @ W[f"{head}.w_frac1"].T + W[f"{head}.b_frac1"]) @ W[f"{head}.w_frac2"].T + W[f"{head}.b_frac2"]
    ).reshape(MAX_OWNED, MAX_PLANETS, N_SHIP_OPTIONS)
    return send, target, frac
'''


V9_BUILD_FEATURES = r'''
def _is_path_blocked_np(src_idx, tgt_idx, positions, radius_arr, is_active):
    src_pos = positions[src_idx]
    tgt_pos = positions[tgt_idx]
    seg = tgt_pos - src_pos
    seg_len_sq = np.sum(seg * seg)
    if seg_len_sq < 1e-9:
        return False
        
    for idx in range(MAX_PLANETS):
        if idx == src_idx or idx == tgt_idx or not is_active[idx]:
            continue
        # Project blocker on segment
        blocker_pos = positions[idx]
        blocker_r = radius_arr[idx]
        to_blocker = blocker_pos - src_pos
        proj = np.dot(to_blocker, seg) / seg_len_sq
        if 0.0 < proj < 1.0:
            closest = src_pos + proj * seg
            dist = np.linalg.norm(blocker_pos - closest)
            if dist <= blocker_r:
                return True
    return False


def _build_features(planets, fleets, player, step, angular_velocity, comet_planet_ids):
    positions = np.zeros((MAX_PLANETS, 2), dtype=np.float32)
    planet_ids_arr = np.full(MAX_PLANETS, -1, dtype=np.int32)
    owners_arr = np.full(MAX_PLANETS, -1, dtype=np.float32)
    ships_arr = np.zeros(MAX_PLANETS, dtype=np.float32)
    radius_arr = np.zeros(MAX_PLANETS, dtype=np.float32)
    production_arr = np.zeros(MAX_PLANETS, dtype=np.float32)
    is_active = np.zeros(MAX_PLANETS, dtype=bool)
    is_comet_arr = np.zeros(MAX_PLANETS, dtype=np.float32)
    is_rotating_arr = np.zeros(MAX_PLANETS, dtype=np.float32)
    comet_set = set(comet_planet_ids) if (comet_planet_ids is not None and len(comet_planet_ids) > 0) else set()

    pa = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    for p in planets:
        pid = int(p[0])
        if pid < MAX_PLANETS:
            pa[pid] = [float(v) for v in p[:7]]
            
    fleets_to_use = fleets if fleets is not None else []
    fa_len = len(fleets_to_use) if len(fleets_to_use) > 0 else 0
    fa = np.zeros((fa_len, 7), dtype=np.float32)
    for i, f in enumerate(fleets_to_use):
        fa[i] = [float(v) for v in f[:7]]

    for p in planets:
        pid = int(p[0])
        if pid >= MAX_PLANETS:
            continue
        owner, x, y, r, ships, prod = float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        planet_ids_arr[pid] = pid
        positions[pid] = [x, y]
        owners_arr[pid] = owner
        ships_arr[pid] = ships
        radius_arr[pid] = r
        production_arr[pid] = prod
        is_active[pid] = True
        is_comet_arr[pid] = 1.0 if pid in comet_set else 0.0
        orb_r = math.hypot(x - CENTER, y - CENTER)
        is_rotating_arr[pid] = 1.0 if (orb_r + r < ROTATION_RADIUS_LIMIT and pid not in comet_set) else 0.0

    is_owned = (owners_arr == player).astype(np.float32)
    is_enemy = ((owners_arr != player) & (owners_arr != -1)).astype(np.float32)
    is_neutral = (owners_arr == -1).astype(np.float32)
    dist_to_sun = np.sqrt((positions[:, 0] - CENTER) ** 2 + (positions[:, 1] - CENTER) ** 2) / 70.71
    pressure = _fleet_pressure(pa, fa, player)

    base_node = [
        is_owned,
        is_enemy,
        is_neutral,
        positions[:, 0] / 100.0,
        positions[:, 1] / 100.0,
        np.minimum(radius_arr / 10.0, 1.0),
        np.minimum(ships_arr / MAX_SHIPS, 1.0),
        np.minimum(production_arr / MAX_PRODUCTION, 1.0),
        is_rotating_arr,
        is_comet_arr,
        dist_to_sun,
        pressure,
    ]
    bucket_node = []
    for pct in (0.50, 0.75, 1.00):
        bucket_ships = np.maximum(1.0, np.floor(pct * ships_arr + 0.5))
        speeds = np.asarray([_fleet_speed(s) for s in bucket_ships], dtype=np.float32)
        bucket_node.extend([
            np.minimum(bucket_ships / MAX_SHIPS, 1.0),
            speeds / 6.0,
            (bucket_ships >= 20.0).astype(np.float32),
        ])
    # Zero out inactive slots for buckets
    active_mask = is_active.astype(np.float32)
    bucket_node = [b * active_mask for b in bucket_node]
    node_features = np.stack(base_node + bucket_node, axis=-1).astype(np.float32)

    future_sight = _extract_planet_future_oracle(pa, fa, player)
    global_features = _global_features_exact(pa, fa, player, step)

    edge_features = np.zeros((MAX_PLANETS, MAX_PLANETS, EDGE_FEATURE_DIM), dtype=np.float32)
    for src in range(MAX_PLANETS):
        if not is_active[src]:
            continue
        for dst in range(MAX_PLANETS):
            if not is_active[dst]:
                continue
            dx = positions[dst, 0] - positions[src, 0]
            dy = positions[dst, 1] - positions[src, 1]
            dist = math.hypot(float(dx), float(dy))
            angle = math.atan2(float(dy), float(dx))
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            sx = positions[src, 0] + cos_a * (radius_arr[src] + 0.1)
            sy = positions[src, 1] + sin_a * (radius_arr[src] + 0.1)
            seg_x = positions[dst, 0] - sx
            seg_y = positions[dst, 1] - sy
            seg_len_sq = max(float(seg_x * seg_x + seg_y * seg_y), 1e-9)
            proj = max(0.0, min(1.0, ((CENTER - sx) * seg_x + (CENTER - sy) * seg_y) / seg_len_sq))
            close_x = sx + proj * seg_x
            close_y = sy + proj * seg_y
            crosses_sun = 1.0 if math.hypot(float(CENTER - close_x), float(CENTER - close_y)) < SUN_RADIUS else 0.0
            rough_dist = math.hypot(float(seg_x), float(seg_y))
            src_ships = float(ships_arr[src])
            tgt_ships = float(ships_arr[dst])
            turns = []
            clears = []
            for pct in (0.50, 0.75, 1.00):
                bucket = max(1.0, math.floor(pct * src_ships + 0.5))
                speed = _fleet_speed(bucket)
                turns.append(min(rough_dist / max(speed, 1e-6) / 100.0, 1.0))
                clears.append(1.0 if bucket >= tgt_ships + 1.0 else 0.0)
            ratio = min(max(src_ships / max(tgt_ships, 1.0), 0.0), 20.0) / 20.0
            edge_features[src, dst] = [
                dx / 100.0,
                dy / 100.0,
                dist / (100.0 * math.sqrt(2.0)),
                sin_a,
                cos_a,
                crosses_sun,
                0.0,
                ratio,
                turns[0],
                turns[1],
                turns[2],
                clears[0],
                clears[1],
                clears[2],
            ]

    owned_nodes = np.full(MAX_OWNED, -1, dtype=np.int32)
    owned = [i for i in range(MAX_PLANETS) if owners_arr[i] == player and is_active[i]]
    for i, idx in enumerate(sorted(owned)[:MAX_OWNED]):
        owned_nodes[i] = idx

    edge_valid_mask = np.zeros((MAX_OWNED, MAX_PLANETS, N_SHIP_OPTIONS), dtype=bool)
    for slot, src in enumerate(owned_nodes):
        if src < 0:
            continue
        src_ships = float(ships_arr[src])
        edge_valid_mask[slot, src, 0] = True
        for tgt in range(MAX_PLANETS):
            if tgt == src or not is_active[tgt] or edge_features[src, tgt, 5] > 0.5:
                continue
            if _is_path_blocked_np(src, tgt, positions, radius_arr, is_active):
                continue
            opts = [max(1.0, round(0.50 * src_ships)), max(1.0, round(0.75 * src_ships)), src_ships]
            for opt_idx, ship_count in enumerate(opts):
                if ship_count >= MIN_LAUNCH_SHIPS and ship_count <= src_ships:
                    edge_valid_mask[slot, tgt, opt_idx] = True

    return (
        node_features,
        edge_features,
        future_sight,
        global_features,
        owned_nodes,
        edge_valid_mask,
        planet_ids_arr,
        positions,
        ships_arr,
        owners_arr,
        radius_arr,
        production_arr,
        is_active,
        is_comet_arr,
    )
'''


CAP_SELECTED_LAUNCHES = r'''
    selected = np.flatnonzero(action_tgt >= 0)
    if selected.shape[0] > 16:
        drop = selected[16:]
        action_tgt[drop] = -1
        action_frac[drop] = 0
'''


def _replace_block(src: str, start_pattern: str, end_pattern: str, replacement: str) -> str:
    m0 = re.search(start_pattern, src, flags=re.MULTILINE)
    if not m0:
        raise RuntimeError(f"Could not find block start: {start_pattern}")
    m1 = re.search(end_pattern, src[m0.start():], flags=re.MULTILINE)
    if not m1:
        raise RuntimeError(f"Could not find block end after {start_pattern}: {end_pattern}")
    start = m0.start()
    end = m0.start() + m1.start()
    return src[:start] + textwrap.dedent(replacement).lstrip() + "\n\n" + src[end:]


def build_main(checkpoint: Path, decode_mode: str, launch_temp: float = 1.0) -> str:
    source = (ROOT / "models" / "planet_transformer_jax" / "agent.py").read_text(encoding="utf-8")
    ckpt_b64 = base64.b64encode(checkpoint.read_bytes()).decode("ascii")

    source = _replace_block(
        source,
        r"^_ROOT = os\.path\.dirname",
        r"^def _silu",
        V9_INIT
            .replace("__CKPT_B64__", ckpt_b64)
            .replace("__DECODE_MODE__", decode_mode)
            .replace("__LAUNCH_TEMP__", str(float(launch_temp))),
    )
    source = _replace_block(source, r"^def _forward", r"^def _obs_to_arrays", V9_FORWARD)
    source = _replace_block(source, r"^def _build_features", r"^def _fleet_speed", V9_BUILD_FEATURES)

    # Apply temperature to send_logits in both decode paths
    # deterministic path
    source = source.replace(
        'send_pair = np.stack([np.zeros_like(send_logits), np.where(has_send, send_logits, -1e9)], axis=-1)',
        'send_pair = np.stack([np.zeros_like(send_logits), np.where(has_send, send_logits / _LAUNCH_TEMP, -1e9)], axis=-1)',
    )
    # stochastic path — launch decision with temperature
    source = source.replace(
        'send_choice = _sample_categorical(np.asarray([0.0, float(send_logits[slot])], dtype=np.float32))',
        'send_choice = _sample_categorical(np.asarray([0.0, float(send_logits[slot]) / _LAUNCH_TEMP], dtype=np.float32))',
    )
    # target and fraction: argmax (greedy) instead of sampling
    source = source.replace(
        '            tgt = _sample_categorical(masked_tgt[slot])',
        '            tgt = int(np.argmax(masked_tgt[slot]))',
    )
    source = source.replace(
        '            frac = _sample_categorical(slot_frac_logits)',
        '            frac = int(np.argmax(slot_frac_logits))',
    )

    marker = "    return _actions_to_moves(\n"
    if marker not in source:
        raise RuntimeError("Could not locate final action return marker")
    source = source.replace(marker, CAP_SELECTED_LAUNCHES + marker, 1)

    compile(source, "main.py", "exec")
    return source


def smoke_test(main_path: Path) -> None:
    code = textwrap.dedent(
        f"""
        import importlib.util
        import pathlib
        p = pathlib.Path({str(main_path)!r})
        spec = importlib.util.spec_from_file_location("v9_submission_main", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        obs = {{
            "player": 0,
            "step": 0,
            "angular_velocity": 0.035,
            "initial_planets": [
                [0, 0, 20.0, 50.0, 3.0, 80.0, 3],
                [1, 1, 80.0, 50.0, 3.0, 80.0, 3],
                [2, -1, 50.0, 20.0, 3.0, 20.0, 2],
            ],
            "planets": [
                [0, 0, 20.0, 50.0, 3.0, 80.0, 3],
                [1, 1, 80.0, 50.0, 3.0, 80.0, 3],
                [2, -1, 50.0, 20.0, 3.0, 20.0, 2],
            ],
            "fleets": [],
            "comets": [],
            "comet_planet_ids": [],
        }}
        moves = mod.agent(obs)
        assert isinstance(moves, list), type(moves)
        for move in moves:
            assert len(move) == 3, move
        print("smoke_moves", moves[:3])
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package GraphTransformerV9 NumPy submission.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default="graph_transformer_v9_final_sample")
    parser.add_argument("--decode", choices=["sample", "deterministic"], default="sample")
    parser.add_argument("--launch-temp", type=float, default=1.0,
                        help="Temperature for launch/hold decision (default 1.0 = unchanged; 0.3 = sharper)")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")

    out_dir = ROOT / args.out_dir
    build_dir = out_dir / args.name
    build_dir.mkdir(parents=True, exist_ok=True)
    main_path = build_dir / "main.py"
    tar_path = out_dir / f"{args.name}.tar.gz"

    main_src = build_main(checkpoint, args.decode, args.launch_temp)
    main_path.write_text(main_src, encoding="utf-8")
    smoke_test(main_path)

    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(main_path, arcname="main.py")
    print(f"Submission archive packaged: {tar_path} ({tar_path.stat().st_size / 1e6:.3f} MB)")

    if args.submit:
        message = args.message or f"GraphTransformerV9 final {args.decode} trueHF H1020 100 H1060 92"
        print(f"Submitting to Kaggle orbit-wars: {message}")
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "orbit-wars", "-f", str(tar_path), "-m", message],
            text=True,
            capture_output=True,
            check=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
