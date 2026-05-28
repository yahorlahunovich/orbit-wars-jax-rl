"""Copy-paste into a Kaggle GPU notebook (one cell per `# ---- Cell N` marker).

Settings: Accelerator GPU T4 x2, Internet ON.
Run cells in order: 1 → 2 → 3 → 4.
"""

# ---- Cell 1: clone repo + install deps --------------------------------------
import importlib, os, subprocess, sys
from pathlib import Path

GITHUB_REPO = "yahorlahunovich/orbit-wars-jax-rl"
GITHUB_TOKEN = ""   # <-- paste the freshly-generated token

WORK_ROOT = Path("/kaggle/working/repo")
RL_DIR = WORK_ROOT / "rl_training_jax"

if WORK_ROOT.exists():
    subprocess.check_call(["git", "-C", str(WORK_ROOT), "pull", "--quiet"])
    print(f"Updated {WORK_ROOT}")
else:
    clone_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
    subprocess.check_call([
        "git", "clone", "--depth=1", "--quiet", clone_url, str(WORK_ROOT)
    ])
    print(f"Cloned -> {WORK_ROOT}")

assert RL_DIR.exists(), f"rl_training_jax missing under {WORK_ROOT}"

try:
    import jax
    devs = jax.devices()
    if not any(d.platform == "gpu" for d in devs):
        raise RuntimeError("jax is CPU-only")
    print(f"jax devices: {devs}")
except Exception as exc:
    print(f"installing jax[cuda12_pip]: {exc}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade", "jax[cuda12_pip]==0.4.30",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
    import jax
    importlib.reload(jax)
    print(f"jax devices after install: {jax.devices()}")

for pkg in ("optax", "flax", "pyyaml"):
    mod = "yaml" if pkg == "pyyaml" else pkg
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["PYTHONPATH"] = f"{RL_DIR}/src" + os.pathsep + TRAIN_ENV.get("PYTHONPATH", "")
TRAIN_ENV["PYTHONUNBUFFERED"] = "1"
print(f"OK — environment ready | RL_DIR={RL_DIR}")


# ---- Cell 2: smoke check ----------------------------------------------------
import os
import subprocess
import sys
from pathlib import Path

RL_DIR = Path("/kaggle/working/repo/rl_training_jax")
if not RL_DIR.exists():
    raise RuntimeError("Run Cell 1 first — repo not found at /kaggle/working/repo")

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["PYTHONPATH"] = f"{RL_DIR}/src" + os.pathsep + TRAIN_ENV.get("PYTHONPATH", "")
TRAIN_ENV["PYTHONUNBUFFERED"] = "1"

os.chdir(RL_DIR)
subprocess.check_call(
    [sys.executable, "-m", "train_ppo", "--config", "configs/smoke_transformer.yaml"],
    env=TRAIN_ENV,
)


# ---- Cell 3: curriculum training (heuristic -> self-play) -------------------
# Logs every 5 updates: mode | heur_wr | W-L-D | mean_ret | env_sps | loss ...
import os
import subprocess
import sys
from pathlib import Path

RL_DIR = Path("/kaggle/working/repo/rl_training_jax")
if not RL_DIR.exists():
    raise RuntimeError("Run Cell 1 first — repo not found at /kaggle/working/repo")

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["PYTHONPATH"] = f"{RL_DIR}/src" + os.pathsep + TRAIN_ENV.get("PYTHONPATH", "")
TRAIN_ENV["PYTHONUNBUFFERED"] = "1"

os.chdir(RL_DIR)
subprocess.check_call(
    [sys.executable, "-m", "train_ppo", "--config", "configs/transformer_curriculum.yaml"],
    env=TRAIN_ENV,
)


# ---- Cell 4: export submission ----------------------------------------------
import os
import subprocess
import sys
from pathlib import Path

RL_DIR = Path("/kaggle/working/repo/rl_training_jax")
if not RL_DIR.exists():
    raise RuntimeError("Run Cell 1 first — repo not found at /kaggle/working/repo")

TRAIN_ENV = os.environ.copy()
TRAIN_ENV["PYTHONPATH"] = f"{RL_DIR}/src" + os.pathsep + TRAIN_ENV.get("PYTHONPATH", "")
TRAIN_ENV["PYTHONUNBUFFERED"] = "1"

CKPT = RL_DIR / "artifacts" / "jax_ppo_curriculum" / "ckpt_last.npz"
assert CKPT.exists(), f"checkpoint missing: {CKPT} — run Cell 3 first"

os.chdir(RL_DIR)
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
