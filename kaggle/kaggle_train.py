"""Copy-paste into a Kaggle GPU notebook (one cell per `# ---- Cell N` marker).

Recommended flow: push this repo to GitHub, clone in Cell 1, train with
curriculum (heuristic -> self-play), export submission.

Settings: Accelerator GPU T4 x2, Internet ON.
"""

# ---- Cell 1: clone repo + install JAX GPU -----------------------------------
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Set your GitHub repo (HTTPS works without SSH keys on Kaggle).
GITHUB_REPO = "https://github.com/yahorlahunovich/orbit-wars-jax-rl.git"
BRANCH = "main"

WORK_ROOT = Path("/kaggle/working/orbit-wars")
RL_DIR = WORK_ROOT / "rl_training_jax"

if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, GITHUB_REPO, str(WORK_ROOT)])
print(f"Cloned -> {WORK_ROOT}")

try:
    import jax
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("CPU-only jax")
    print(f"jax devices: {jax.devices()}")
except Exception as exc:
    print(f"Installing jax GPU: {exc}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "jax[cuda12_pip]==0.4.30",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])

for pkg in ("optax", "flax", "pyyaml"):
    try:
        __import__(pkg if pkg != "pyyaml" else "yaml")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["PYTHONPATH"] = f"{RL_DIR}/src" + os.pathsep + TRAIN_ENV.get("PYTHONPATH", "")
TRAIN_ENV["PYTHONUNBUFFERED"] = "1"


# ---- Cell 2: smoke check ----------------------------------------------------
os.chdir(RL_DIR)
subprocess.check_call(
    [sys.executable, "-m", "train_ppo", "--config", "configs/smoke_transformer.yaml"],
    env=TRAIN_ENV,
)


# ---- Cell 3: curriculum training (heuristic -> self-play) -------------------
# Logs every 5 updates: mode, heuristic win rate, W-L-D, mean_ret, env_sps, losses.
# Checkpoints every 100 updates under artifacts/jax_ppo_curriculum/.
subprocess.check_call(
    [sys.executable, "-m", "train_ppo", "--config", "configs/transformer_curriculum.yaml"],
    env=TRAIN_ENV,
)


# ---- Cell 4: export submission ----------------------------------------------
CKPT = RL_DIR / "artifacts" / "jax_ppo_curriculum" / "ckpt_last.npz"
assert CKPT.exists(), f"checkpoint missing: {CKPT}"

subprocess.check_call(
    [
        sys.executable, "scripts/export_jax_submission.py",
        "--checkpoint", str(CKPT),
        "--config", "configs/transformer_curriculum.yaml",
        "--output", "/kaggle/working/submission_jax.zip",
    ],
    env=TRAIN_ENV,
)

print("Done. Download /kaggle/working/submission_jax.zip from the Output tab.")
