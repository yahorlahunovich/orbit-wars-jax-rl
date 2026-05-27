"""Smoke-test a submission_jax.zip locally before uploading it to Kaggle.

Unzips the bundle into a temp dir, runs the agent against a couple of toy
observations, and reports per-call latency. Useful after exporting a new
checkpoint to catch packaging mistakes early.

Usage:

    conda run -n ml python kaggle/test_submission_locally.py \
        --submission submission_jax.zip
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(args.submission, "r") as zf:
            zf.extractall(td_path)
        sub_dir = td_path / "submission_jax"
        assert (sub_dir / "main.py").exists(), f"main.py missing in submission: {sub_dir}"

        # Mirror Kaggle's import flow: cwd = submission dir, sys.path[0] = '.'.
        import os
        prev_cwd = os.getcwd()
        os.chdir(sub_dir)
        sys.path.insert(0, str(sub_dir))
        try:
            main_mod = importlib.import_module("main")
            agent = main_mod.agent

            obs_basic = {
                "player": 0, "step": 5, "angular_velocity": 0.02,
                "planets": [
                    [0, 0, 20.0, 20.0, 4.0, 50.0, 3.0],
                    [1, 1, 80.0, 80.0, 4.0, 30.0, 2.0],
                    [2, -1, 30.0, 70.0, 3.0, 5.0, 1.0],
                    [3, 0, 70.0, 30.0, 4.0, 20.0, 2.0],
                ],
                "fleets": [],
                "comets": [],
                "comet_planet_ids": [],
                "initial_planets": [
                    [0, 0, 20.0, 20.0, 4.0, 50.0, 3.0],
                    [1, 1, 80.0, 80.0, 4.0, 30.0, 2.0],
                    [2, -1, 30.0, 70.0, 3.0, 5.0, 1.0],
                    [3, 0, 70.0, 30.0, 4.0, 20.0, 2.0],
                ],
                "next_fleet_id": 0,
            }

            t0 = time.perf_counter()
            moves = agent(obs_basic)
            t1 = time.perf_counter()
            print(f"first call (JIT compile): {(t1-t0)*1000:.0f} ms")
            print(f"  moves: {moves}")

            assert isinstance(moves, list)
            for m in moves:
                assert len(m) == 3
                assert isinstance(m[2], int)

            for _ in range(3):
                _ = agent(obs_basic)
            t2 = time.perf_counter()
            N = 30
            for _ in range(N):
                _ = agent(obs_basic)
            t3 = time.perf_counter()
            print(f"steady-state: {(t3-t2)/N*1000:.2f} ms / call")

            # Empty board → should return [] without crashing.
            empty_obs = dict(obs_basic, planets=[], initial_planets=[])
            moves_empty = agent(empty_obs)
            assert moves_empty == []
            print("empty board OK")

            print("\nSUBMISSION OK.")
        finally:
            os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
