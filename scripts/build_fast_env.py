#!/usr/bin/env python3
"""Build a local patched kaggle_environments tree with a Numba Orbit Wars hot loop."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMPORT_MARKER = "import random\n"
IMPORT_PATCH = (
    "import random\n"
    "import numpy as np\n"
    "from kaggle_environments.envs.orbit_wars.fast_orbit_core import (\n"
    "    generate_comet_paths_fast,\n"
    "    move_fleets_core_numba,\n"
    "    warm_numba,\n"
    ")\n"
)

DISTANCE_MARKER = '''def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
'''

DISTANCE_PATCH = '''warm_numba()


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
'''

OLD_BLOCK = '''    # 3. Fleet Movement (with continuous swept-pair collision detection)
    # Speed scales with fleet size: 1 ship = 1/turn, max = shipSpeed (default 6)
    max_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # Check if fleet path intersected any planet (continuous collision).
        # Check planets first so fast fleets that would overshoot the bounds
        # or sun still get credit for hitting a planet along the way.
        hit_planet = False
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:
                continue
            p_old, p_new, _ = path
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue

        # Check if fleet went out of bounds
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue

        # Check if fleet path crossed the sun
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue
'''

NEW_BLOCK = '''    # 3. Fleet Movement (with continuous swept-pair collision detection)
    # Speed scales with fleet size: 1 ship = 1/turn, max = shipSpeed (default 6)
    max_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    if obs0.fleets:
        fleet_arr = np.asarray(obs0.fleets, dtype=np.float64).reshape((-1, 7))
        planet_path_rows = []
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None:
                continue
            p_old, p_new, check_collision = path
            planet_path_rows.append(
                [
                    float(planet[0]),
                    float(p_old[0]),
                    float(p_old[1]),
                    float(p_new[0]),
                    float(p_new[1]),
                    float(planet[4]),
                    1.0 if check_collision else 0.0,
                ]
            )
        planet_path_arr = np.asarray(planet_path_rows, dtype=np.float64).reshape((-1, 7))
        new_xy, remove_mask, hit_planet_index = move_fleets_core_numba(
            fleet_arr, planet_path_arr, float(max_speed)
        )

        for i, fleet in enumerate(obs0.fleets):
            fleet[2] = float(new_xy[i, 0])
            fleet[3] = float(new_xy[i, 1])
            hit_idx = int(hit_planet_index[i])
            if hit_idx >= 0:
                planet_id = int(planet_path_arr[hit_idx, 0])
                combat_lists[planet_id].append(fleet)
            if bool(remove_mask[i]):
                fleets_to_remove.append(fleet)
'''

FAST_COMET_FUNCTION = '''def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    """Fast Numba-backed equivalent of the official comet path generator."""
    return generate_comet_paths_fast(
        initial_planets,
        angular_velocity,
        spawn_step,
        comet_planet_ids,
        comet_speed,
        rng,
    )


'''


def copy_env_tree(source_root: Path, output_root: Path) -> None:
    src_pkg = source_root / "kaggle_environments"
    dst_pkg = output_root / "kaggle_environments"
    if not src_pkg.exists():
        raise FileNotFoundError(f"Missing source package: {src_pkg}")
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(
        src_pkg,
        dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache"),
    )


def patch_orbit_wars(output_root: Path, core_path: Path) -> None:
    orbit_dir = output_root / "kaggle_environments" / "envs" / "orbit_wars"
    target = orbit_dir / "orbit_wars.py"
    if not target.exists():
        raise FileNotFoundError(target)

    shutil.copy2(core_path, orbit_dir / "fast_orbit_core.py")

    text = target.read_text(encoding="utf-8")
    if "fast_orbit_core import move_fleets_core_numba" not in text:
        if IMPORT_MARKER not in text:
            raise RuntimeError("Could not find import patch marker")
        text = text.replace(IMPORT_MARKER, IMPORT_PATCH, 1)
    if "warm_numba()\n\n\ndef distance" not in text:
        if DISTANCE_MARKER not in text:
            raise RuntimeError("Could not find warm patch marker")
        text = text.replace(DISTANCE_MARKER, DISTANCE_PATCH, 1)

    if "Fast Numba-backed equivalent of the official comet path generator" not in text:
        start = text.find("def generate_comet_paths(")
        end = text.find("\ndef interpreter(", start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not find comet generator block to patch")
        text = text[:start] + FAST_COMET_FUNCTION + text[end + 1 :]

    if NEW_BLOCK not in text:
        if OLD_BLOCK not in text:
            raise RuntimeError("Could not find fleet movement block to patch")
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)

    target.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/media/yahor/ADATA SE880/datasets/kaggle-environments-master"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/fast_kaggle_env"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    core_path = repo_root / "scripts" / "fast_orbit_core.py"

    copy_env_tree(args.source_root, output_root)
    patch_orbit_wars(output_root, core_path)

    print(f"Wrote patched environment root: {output_root}")
    print("Use with:")
    print(f"  --candidate-env-root {output_root}")


if __name__ == "__main__":
    main()
