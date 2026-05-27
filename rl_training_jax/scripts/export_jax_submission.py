"""Package a trained JAX policy into a Kaggle-ready submission zip.

Usage (from `rl_training_jax/`):

    python scripts/export_jax_submission.py \
        --checkpoint artifacts/jax_ppo_transformer/ckpt_last.npz \
        --config configs/transformer_selfplay.yaml \
        --output ../submission_jax.zip

What it does:

1. Loads the checkpoint (`.npz` produced by `train_ppo.py`).
2. Re-serializes the flax params into `weights/policy.msgpack`.
3. Writes `weights/model_config.json` with d_model/n_heads/etc.
4. Copies a minimal subset of `orbit_wars/` + `policy.py` into
   `submission_jax/src/`.
5. Zips the `submission_jax/` directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit_wars import FLEET_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_FLEETS, MAX_PLANETS, PLANET_FEATURE_DIM
from policy import PlanetPolicy
from train_ppo import load_config


# These are the files actually needed at inference time. Training-only files
# (env, reset, step, comet, reference) are excluded to keep the submission lean
# and free of any kaggle_environments dependency.
INFERENCE_FILES = [
    "__init__.py",
    "constants.py",
    "state.py",
    "geometry.py",
    "convert.py",
    "features_jax.py",
    "decode.py",
]


def _filter_init_imports(text: str) -> str:
    """Strip imports of training-only modules from `orbit_wars/__init__.py`
    so the submission package doesn't need env/step/reset/comet/reference."""
    drop_lines = (
        "from .env import",
        "from .reference import",
        "from .reset import",
        "from .step import",
        "from .comet import",
    )
    out = []
    for line in text.splitlines(keepends=True):
        if any(line.startswith(d) for d in drop_lines):
            continue
        out.append(line)
    # Also remove these symbols from __all__.
    text = "".join(out)
    for sym in ("OrbitWarsJaxEnv", "VectorOrbitWarsEnv", "reset", "step", "step_jit",
                "batched_step", "reference_reset", "reference_step"):
        text = text.replace(f'    "{sym}",\n', "")
    return text


def export(checkpoint: Path, config: Path, output: Path) -> None:
    cfg = load_config(config)
    submission_dir = ROOT.parent / "submission_jax"
    src_dir = submission_dir / "src"
    weights_dir = submission_dir / "weights"
    pkg_dir = src_dir / "orbit_wars"

    # Clean & recreate.
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Copy inference files.
    src_pkg = ROOT / "src" / "orbit_wars"
    for fname in INFERENCE_FILES:
        text = (src_pkg / fname).read_text(encoding="utf-8")
        if fname == "__init__.py":
            text = _filter_init_imports(text)
        (pkg_dir / fname).write_text(text, encoding="utf-8")

    # Copy policy.py.
    shutil.copy2(ROOT / "src" / "policy.py", src_dir / "policy.py")

    # Load checkpoint and re-serialize params.
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    ckpt = np.load(checkpoint, allow_pickle=False)
    blob = bytes(ckpt["params"].tobytes())

    # Sanity: round-trip through flax with the right model shape to validate.
    model = PlanetPolicy(
        planet_count=MAX_PLANETS, fleet_count=MAX_FLEETS,
        d_model=cfg.d_model, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, bucket_count=cfg.bucket_count,
    )
    example = {
        "planet_features": jnp.zeros((1, MAX_PLANETS, PLANET_FEATURE_DIM), jnp.float32),
        "planet_mask": jnp.ones((1, MAX_PLANETS), jnp.bool_),
        "fleet_features": jnp.zeros((1, MAX_FLEETS, FLEET_FEATURE_DIM), jnp.float32),
        "fleet_mask": jnp.ones((1, MAX_FLEETS), jnp.bool_),
        "global_features": jnp.zeros((1, GLOBAL_FEATURE_DIM), jnp.float32),
    }
    init_params = model.init(jax.random.PRNGKey(0), **example)
    params = flax.serialization.from_bytes(init_params, blob)
    _ = model.apply(params, **example)            # forward smoke
    blob = flax.serialization.to_bytes(params)

    (weights_dir / "policy.msgpack").write_bytes(blob)
    (weights_dir / "model_config.json").write_text(
        json.dumps({
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "bucket_count": cfg.bucket_count,
            "planet_feature_dim": PLANET_FEATURE_DIM,
            "fleet_feature_dim": FLEET_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
        }, indent=2),
        encoding="utf-8",
    )

    # Build the zip.
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in submission_dir.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(submission_dir.parent)
                zf.write(path, arcname=arcname)

    print(f"Wrote submission: {output}")
    print(f"Submission size: {output.stat().st_size / 1024:.1f} KiB")
    print(f"Weights bytes:   {len(blob)} ({len(blob)/1024:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default="../submission_jax.zip", type=Path)
    args = parser.parse_args()
    export(args.checkpoint.resolve(), args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
