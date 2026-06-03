import jax
import jax.numpy as jnp
import time
from rl_training_jax.src.train_ppo import train, load_config

cfg = load_config("rl_training_jax/configs/smoke_transformer.yaml")
t0 = time.time()
train(cfg)
print(f"Total time: {time.time() - t0:.2f}s")
