"""Copy-paste this whole file into a Kaggle GPU notebook to train the
JAX Transformer PPO policy and export a ready-to-submit zip.

Prerequisites (set up in the Kaggle notebook UI):

1. Notebook -> Settings -> Accelerator: GPU T4 x2 (or P100 — either works).
2. Notebook -> Settings -> Internet: ON (only needed for the first cell that
   pip-installs jax[cuda]).
3. Upload this whole repository as a Kaggle Dataset. Recommended dataset
   name: `orbit-wars-rl-template`. After attaching it to the notebook, the
   files appear at `/kaggle/input/orbit-wars-rl-template/`.

The script then:

- Copies the source tree to a writable scratch dir (`/kaggle/working/repo`).
- Installs the GPU build of JAX if not already present.
- Runs PPO with `configs/transformer_selfplay.yaml`.
- Exports `submission_jax.zip` to `/kaggle/working/` so you can download it.
"""

# ---- Cell 1: environment setup ---------------------------------------------
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_NAME = "orbit-wars-rl-template"          # change if you named yours differently
DATASET_ROOT = Path(f"/kaggle/input/{DATASET_NAME}")
WORK_ROOT = Path("/kaggle/working/repo")
RL_DIR = WORK_ROOT / "rl_training_jax"

assert DATASET_ROOT.exists(), (
    f"Dataset not found at {DATASET_ROOT}. Attach your repo as a Kaggle Dataset "
    "and update DATASET_NAME above to match its slug."
)
if not WORK_ROOT.exists():
    shutil.copytree(DATASET_ROOT, WORK_ROOT)
    print(f"Copied dataset -> {WORK_ROOT}")
else:
    print(f"Reusing existing {WORK_ROOT}")

# Install JAX GPU build if needed. Kaggle usually has CPU jax pre-installed.
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
    # Force re-import after install.
    import importlib
    import jax
    importlib.reload(jax)
    print(f"jax devices after install: {jax.devices()}")

# Make sure optax, flax, pyyaml are present (they normally are on Kaggle).
for pkg in ("optax", "flax", "pyyaml"):
    try:
        __import__(pkg if pkg != "pyyaml" else "yaml")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


# ---- Cell 2: smoke check (recommended before launching a long run) ----------
os.chdir(RL_DIR)
sys.path.insert(0, str(RL_DIR / "src"))

# Tiny smoke to confirm the GPU build, env, policy, and PPO all work.
subprocess.check_call([
    sys.executable, "-m", "train_ppo",
    "--config", "configs/smoke_transformer.yaml",
])


# ---- Cell 3: real training run on GPU --------------------------------------
# Uses configs/transformer_selfplay.yaml. Tune `total_updates` to your budget.
subprocess.check_call([
    sys.executable, "-m", "train_ppo",
    "--config", "configs/transformer_selfplay.yaml",
])


# ---- Cell 4: export submission zip -----------------------------------------
CKPT = RL_DIR / "artifacts" / "jax_ppo_transformer" / "ckpt_last.npz"
assert CKPT.exists(), f"checkpoint missing: {CKPT}"

subprocess.check_call([
    sys.executable, "scripts/export_jax_submission.py",
    "--checkpoint", str(CKPT),
    "--config", "configs/transformer_selfplay.yaml",
    "--output", "/kaggle/working/submission_jax.zip",
])

print("Submission ready at /kaggle/working/submission_jax.zip")
print("Use the right-hand 'Output' tab to download it.")
