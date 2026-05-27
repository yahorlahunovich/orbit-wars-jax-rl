# JAX Orbit Wars Submission

This directory is **assembled by** `scripts/export_jax_submission.py`. The
script copies:

- `rl_training_jax/src/orbit_wars/` (minus `reference.py`, `env.py`, `reset.py`,
  `step.py`, `comet.py` which are only needed for training)
- `rl_training_jax/src/policy.py`
- The flax-serialized weights into `weights/policy.msgpack`
- A small `weights/model_config.json` capturing the model architecture used
  during training.

Then it produces `submission_jax.zip` in the repo root, ready to upload to
Kaggle.

Do not edit files in this directory manually — rerun the export script.
