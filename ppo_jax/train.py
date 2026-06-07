import argparse
import jax
import yaml
import time
import os
import equinox as eqx
from pathlib import Path
from src.ppo import make_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default_cfg.yaml")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    ppo_data = data.get("ppo", {})
    env_data = data.get("env", {})
    model_data = data.get("model", {})
    
    num_envs = int(ppo_data.get("num_envs", env_data.get("num_envs", 4)))
    rollout_steps = int(ppo_data.get("rollout_steps", env_data.get("rollout_steps", 128)))
    total_updates = int(ppo_data.get("total_updates", 500))
    
    flat_config = {
        "LR": float(ppo_data.get("lr", ppo_data.get("pi_lr", 2.5e-4))),
        "NUM_ENVS": num_envs,
        "NUM_STEPS": rollout_steps,
        "TOTAL_TIMESTEPS": int(total_updates * rollout_steps * num_envs),
        "UPDATE_EPOCHS": int(ppo_data.get("epochs", ppo_data.get("train_pi_iters", 4))),
        "NUM_MINIBATCHES": int(ppo_data.get("num_minibatches", 4)),
        "GAMMA": float(ppo_data.get("gamma", 0.99)),
        "GAE_LAMBDA": float(ppo_data.get("gae_lambda", 0.95)),
        "CLIP_EPS": float(ppo_data.get("clip_coef", 0.2)),
        "ENT_COEF": float(ppo_data.get("ent_coef", 0.01)),
        "VF_COEF": float(ppo_data.get("vf_coef", 0.5)),
        "MAX_GRAD_NORM": float(ppo_data.get("max_grad_norm", 0.5)),
        "ANNEAL_LR": True,
        
        # Env specific
        "EPISODE_STEPS": int(env_data.get("episode_steps", 500)),
        "SHIP_SPEED": float(env_data.get("ship_speed", 6.0)),
        
        # Model specific
        "D_MODEL": int(model_data.get("hidden_size", model_data.get("d_model", 64))),
        "NUM_HEADS": int(model_data.get("num_heads", 4)),
        "NUM_LAYERS": int(model_data.get("num_layers", 5)),
    }
    return flat_config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"Starting JAX PPO training with config: {config}")
    
    rng = jax.random.PRNGKey(42)
    init_fn, update_fn = make_train(config)
    
    print("Compiling network and initial environment state...")
    runner_state = init_fn(rng)
    
    print("JIT Compiling update step...")
    update_fn_jit = jax.jit(update_fn)
    
    # Warmup compilation run
    runner_state, metrics = update_fn_jit(runner_state)
    jax.block_until_ready(metrics)
    print("Compilation finished. Starting training loop.")
    
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    
    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()
    
    for update_idx in range(1, num_updates + 1):
        runner_state, metrics = update_fn_jit(runner_state)
        jax.block_until_ready(metrics)
        
        if update_idx % 5 == 0 or update_idx == 1:
            elapsed = time.time() - start_time
            sps = (update_idx * steps_per_update) / elapsed
            print(f"Update {update_idx:04d}/{num_updates} | SPS: {sps:.1f}")
            print(f"  Reward:     {metrics['reward']:.3f}")
            print(f"  Loss:       {metrics['loss']:.3f} (P: {metrics['policy_loss']:.3f}, V: {metrics['value_loss']:.3f}, E: {metrics['entropy']:.3f})")
            print(f"  Approx KL:  {metrics['approx_kl']:.4f}")
            print(f"  Clip Frac:  {metrics['clip_frac']:.4f}")
            print(f"  Expl. Var:  {metrics['explained_variance']:.4f}")
            print("-" * 50)
            
        if update_idx % 100 == 0:
            train_state = runner_state[0]
            ckpt_path = f"checkpoints/ckpt_{update_idx:04d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, train_state.model)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final model
    train_state = runner_state[0]
    final_path = "checkpoints/model_final.eqx"
    eqx.tree_serialise_leaves(final_path, train_state.model)
    print(f"Training finished! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
