# Orbit Wars JAX PPO Training Pipeline Guide

This document provides a highly detailed, end-to-end explanation of the **JAX/Flax Transformer PPO** reinforcement learning pipeline implemented in `rl_training_jax/`.

---

## 1. System Overview

The pipeline utilizes a hybrid approach: environment rollout stepping is written in pure Python (using host-side resets and comet generation), but the core simulation step, feature extraction, policy forward pass, and gradient updates are implemented in **JAX** and JIT-compiled to execute at maximum throughput.

```mermaid
graph TD
    A[Python Env Manager] -->|Host resets & Comet spawn| B[Padded States]
    B -->|features_jax.py| C[Padded Planet Features (B, P, 58)]
    C -->|Transformer Policy| D[Target & Bucket Logits]
    D -->|decode.py / rollout.py| E[Sample target O(P^2) -> Sample bucket O(P*B)]
    E -->|step_jit| F[Advance Simulation]
    F -->|Potential-Based Shaping| G[Shaped Rewards]
    G -->|compute_gae| H[Advantages & Returns]
    H -->|PPO Joint Loss| I[Optax Parameter Update]
```

---

## 2. JAX State Representation & Conversion

JAX requires static array shapes to avoid recompilation. Thus, the entire state of the game is padded up to fixed maximum dimensions.

### 2.1 State Structure (`state.py`)
The game state is managed by the `OrbitWarsState` struct:
* **`planets` (shape `[96, 8]`)**: Features per planet: `[id, owner, x, y, radius, ships, production, active_flag]`. Pad planets have `active_flag = 0.0`.
* **`initial_planets` (shape `[96, 8]`)**: The state of planets at step 0, used to compute orbit tracking since angular coordinates rotate over time.
* **`fleets` (shape `[256, 8]`)**: In-flight fleet details: `[id, owner, x, y, angle, from_planet_id, ships, active_flag]`.
* **`comets`**: A nested `CometGroups` struct storing positions, paths (up to 64 steps), active flags, and path indices for comet groups.

### 2.2 Host-Side State Synchronization (`convert.py`)
To interface with the Python-based Kaggle environment:
1. **`observation_to_state`**: Parses the list-based JSON observation dict from Kaggle, pads the planets array up to $96$ slots and fleets up to $256$ slots, and builds the JAX-native `OrbitWarsState`.
2. **`state_to_observation_dict`**: Extracts active entries from the padded JAX arrays, strips padding, and reconstructs the standard Kaggle environment dictionary.

---

## 3. Feature Engineering & Normalization (`features_jax.py`)

Each step, the batched states are passed through a player-relative JAX feature encoder. Every planet feature vector is **58-dimensional** and normalized between $[0, 1]$ or $[-1, 1]$.

### 3.1 Per-Planet Local Features (13 dimensions)
* **0–2: Ownership**: One-hot indicators for planet ownership (`is_mine`, `is_enemy`, `is_neutral`).
* **3: my_ships_norm**: $\log(1 + \text{ships}) / \log(1 + 5000)$ (only if owned by learner).
* **4: enemy_ships_norm**: $\log(1 + \text{ships}) / \log(1 + 5000)$ (only if owned by opponent).
* **5: production_norm**: $\text{production} / 5.0$.
* **6: radius_norm**: $\text{radius} / 10.0$.
* **7–8: Position**: Planet coordinates normalized as $x / 100.0$ and $y / 100.0$.
* **9–10: Orbit features**: $\sin(\text{angle})$ and $\cos(\text{angle})$ for rotating planets (static comets/planets are set to $0.0$).
* **11: orbit_r_norm**: Orbit radius $/ 50.0$.
* **12: is_comet**: Binary indicator $1.0$ if the planet is a comet, $0.0$ otherwise.

### 3.2 Tactical & In-Transit Fleet Features (11 dimensions)
Since fleets are in transit, we project their trajectory:
* **13: comet_remaining_norm**: Remaining steps of comet life (before expiry) $/ 64.0$.
* **14–15: incoming_ships_norm**: Aggregated incoming ship counts for me and the enemy: $\log(1 + \text{ships}) / \log(1 + 500)$.
* **16–17: incoming_eta_norm**: ETA of the nearest incoming fleet: $\min(\text{eta}, 100) / 100.0$.
* **18: nearest_enemy_dist**: Normalized distance to nearest enemy planet $/ 50.0$.
* **19: nearest_friend_dist**: Distance to nearest owned planet $/ 50.0$.
* **20: nearest_enemy_ships**: Ships on the nearest enemy planet (log-normalized).
* **21: roi_norm**: Economic utility score: $(\text{production} / (\text{ships} + 1.0)) / 2.0$.
* **22: is_high_value**: Binary flag if $\text{production} \ge 3$.
* **23: nearest_enemy_time**: Travel time to the nearest enemy planet at maximum ship speed $/ 100.0$.

### 3.3 Broadcasted Global Context Features (34 dimensions)
To give the Transformer token attention global awareness, 34 global stats are broadcasted directly into every planet token:
* **24–26: Global planet count ratios**: own count, enemy count, neutral count (all $/ 12.0$).
* **27–28: Total garrisoned ships**: own count and enemy count (log-normalized).
* **29–30: Total fleet ships**: Own and enemy fleet ship counts (log-normalized) currently in transit.
* **31–32: Global production**: Own total production and enemy total production (both $/ 50.0$).
* **33: turn**: $\text{current\_step} / \text{episode\_steps}$.
* **34: is_late_game**: $1.0$ if $\text{turn} > 0.5$.
* **35: active_comet_count**: Count of active comets $/ 8.0$.
* **36–37: Largest planets**: Garrison size of own and enemy largest planets.
* **38–39: Max production**: Maximum production value on own and enemy planets.
* **40: next_comet_eta**: Turn delta to the next comet spawn step $/ 500.0$.
* **41: prod_lead**: Relative production lead: $(\text{my\_prod} - \text{opp\_prod}) / (\text{my\_prod} + \text{opp\_prod} + 1.0)$.
* **42: ship_lead**: Relative ship lead: $(\text{my\_ships} - \text{opp\_ships}) / (\text{my\_ships} + \text{opp\_ships} + 1.0)$.
* **43–44: Global fleet count**: Total own and enemy fleet objects active.
* **45: ship_rank_all**: Normalized rank of this planet's ships compared to all active planets.
* **46: prod_rank_all**: Rank of this planet's production compared to all active planets.
* **47: my_ship_rank**: Rank of this planet's ships compared to own planets only.
* **48: enemy_ship_rank**: Rank of this planet's ships compared to enemy planets only.
* **49: is_my_largest**: $1.0$ if this is my largest planet by ship count.
* **50: is_enemy_largest**: $1.0$ if this is the enemy's largest planet by ship count.
* **51: would_lose**: $1.0$ if incoming enemy ships exceed current garrison + my incoming reinforcements.
* **52: net_balance_log**: Signed log ratio of total garrison + reinforcements - incoming threats.

### 3.4 Expert Predictive Features (5 dimensions)
* **53–54: Projected Garrisons**: Garrison size projected 10 and 20 steps into the future: $\log(1 + \text{proj\_ships}) / \log(1 + 5000)$.
* **55: slack**: Garrison ships available to launch without risking losing the planet to projected threats.
* **56: is_base**: $1.0$ if this is the learner's starting base planet.
* **57: is_enemy_base**: $1.0$ if this is the opponent's starting base planet.

---

## 4. Two-Stage Action Space & Masking (`decode.py`)

A full action space representing source planets, target planets, and ship buckets has size $P \times P \times B$. Evaluating this leads to expensive $96 \times 96 \times 4 = 36,864$ tensor operations. 

To prevent this bottleneck, the pipeline splits action decoding and masking into **two stages**:

```
[Target Phase] 
For all sources (P), evaluate all targets (P) -> Score O(P^2)
        ↓
  Sample target index (target_idx)
        ↓
[Bucket Phase]
For all sources (P), evaluate ONLY the chosen target -> Score O(P * B)
        ↓
  Sample bucket index (bucket_idx)
```

### 4.1 Action Masking Constraints
During the Target Phase, `compose_target_grid` generates a boolean mask `target_mask` of shape `(B, P, P)` enforcing:
1. **Ownership**: Source planet must be owned by the active player.
2. **Min Launch**: Source planet must contain $\ge 5$ ships.
3. **Obstacles**: 
   * **Sun Collision**: Path must not pass through the Sun (`SUN_RADIUS = 10.0` with `margin = 1.5`).
   * **Planet Obstruction**: Path must not collide with other active planets (`planet_radius + margin`).

During the Bucket Phase, `compose_bucket_grid` creates a `bucket_valid` mask of shape `(B, P, 4)` checking:
* The source has enough ships to satisfy the bucket fraction without falling below the launch minimum.

### 4.2 Ship Buckets
We evaluate 4 buckets corresponding to the fraction of ships to send:
* **Bucket 0**: $25\%$ of source garrison.
* **Bucket 1**: $50\%$ of source garrison.
* **Bucket 2**: $75\%$ of source garrison.
* **Bucket 3**: $100\%$ of source garrison.

All values are floored and clipped to be $\ge 5$ ships.

---

## 5. Policy Network Architecture (`policy.py`)

The policy utilizes a **Transformer encoder** over planet tokens, sharing the representation backbone between the actor and the critic.

```
Input: Planet Features (B, 96, 58) & Planet Mask (B, 96)
  ↓
Linear Embedding (58 -> d_model) + LayerNorm + GeLU
  ↓
3 x Transformer Blocks (num_heads=4, ff_mult=4) with Padding Mask
  ↓
Planet representations (B, 96, d_model)
  ├── Actor Target Head: Query-Key Dot-Product Attention -> Target Logits (B, 96, 96)
  ├── Actor Bucket Head: Source-Target Factorized MLP -> Bucket Logits (B, 96, 96, 4)
  └── Critic Value Head: Mean-Pooling -> MLP -> State Value (B, 1)
```

### 5.1 Actor Heads
* **Target Head**: Projects representations to queries $Q$ and keys $K$. The raw logits are computed via Einstein summation:
  $$\text{target\_logits} = \frac{Q K^T}{\sqrt{d_{model}}}$$
  A learnable `noop_bias` is added to the diagonal ($Q_i K_i^T$) to naturally parameterize the NOOP action (attack/reinforce self).
* **Bucket Head**: Projects source and target representations separately to bucket logits and combines them additively:
  $$\text{bucket\_logits}_{i, j} = \text{MLP}_{src}(h_i) + \text{MLP}_{tgt}(h_j)$$

### 5.2 Critic Head
Performs masked mean-pooling across the active planet tokens to obtain a global state representation, then passes it through a 2-layer MLP to output a scalar state value $V(s)$.

---

## 6. GAE, PPO Loss & Optimization (`ppo.py`, `train_ppo.py`)

### 6.1 Generalized Advantage Estimation (GAE)
GAE is computed in reverse time scan (`compute_gae`):
$$\delta_t = R^{shaped}_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
$$A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}$$
where $d_t$ is the done flag, $\gamma$ is the discount factor, and $\lambda$ is the GAE parameter. The return is computed as $Return_t = A_t + V(s_t)$.

### 6.2 Joint Loss Formulation
PPO updates parameters using a joint loss function:
$$L_{joint} = L_{policy} + c_{value} L_{value} - c_{entropy} H$$
* **Policy Loss**: Clipped ratio surrogate loss, masked to ignore sources with no valid targets:
  $$r_t(\theta) = \exp(\log \pi_\theta(a|s) - \log \pi_{old}(a|s))$$
  $$L_{policy} = -\mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$
* **Value Loss**: Mean Squared Error between critic output and returns.
* **Entropy Bonus ($H$)**: Average joint entropy of target selection and bucket selection:
  $$H = H(\pi_{target}) + H(\pi_{bucket}|\text{chosen\_target})$$

### 6.3 Learning Rate Schedules
The learning rates for the policy head (`pi`) and value head (`vf`) are managed separately by `optax.warmup_cosine_decay_schedule` using warm-up steps and cosine decay to a minimum learning rate (`lr_end`).

---

## 7. Potential-Based Reward Shaping

To solve the sparse reward problem without distorting the optimal policy, the pipeline uses **potential-based reward shaping** derived from game states.

### 7.1 Mathematical Form
The shaped reward is defined as:
$$R^{shaped}_t = \begin{cases} \gamma \Phi(s_{t+1}) - \Phi(s_t) & \text{if } t < T \\ R^{raw}_T - \Phi(s_T) & \text{if } t = T \text{ (terminal)} \end{cases}$$
Because the potentials telescope over time, the discounted sum of shaped rewards is:
$$\sum_{t=0}^T \gamma^t R^{shaped}_t = \sum_{t=0}^T \gamma^t R^{raw}_t - \Phi(s_0)$$
Since $\Phi(s_0)$ depends only on the initial state, the reward shaping does not alter the optimal policy.

### 7.2 Shaping Potential $\Phi(s)$
The state potential is evaluated as:
$$\Phi(s) = 0.10 \times \frac{\text{my\_planets} - \text{opp\_planets}}{\text{total\_planets}}$$
$$+ 0.05 \times \text{clip}\left(\frac{\text{my\_production} - \text{opp\_production}}{10.0}, -1.0, 1.0\right)$$
$$+ 0.02 \times \text{clip}\left(\frac{\text{my\_total\_ships} - \text{opp\_total\_ships}}{100.0}, -1.0, 1.0\right)$$

---

## 8. Hyperparameter Reference

| Parameter | Smoke Transformer Config | Self-Play Transformer Config |
| :--- | :--- | :--- |
| **`seed`** | `0` | `0` |
| **`num_envs`** | `8` | `128` |
| **`episode_steps`** | `200` | `500` |
| **`rollout_steps`** | `16` | `32` |
| **`d_model`** | `48` | `96` |
| **`num_heads`** | `4` | `4` |
| **`num_layers`** | `2` | `3` |
| **`bucket_count`** | `4` | `4` |
| **`total_updates`** | `20` | `5000` |
| **`epochs`** (`train_pi_iters`) | `2` | `1` |
| **`minibatch_size`** | `256` | `1024` |
| **`gamma`** ($\gamma$) | `0.9999` | `0.99` |
| **`gae_lambda`** ($\lambda$) | `0.95` | `0.95` |
| **`clip_coef`** ($\epsilon$) | `0.2` | `0.2` |
| **`ent_coef`** ($c_{entropy}$) | `0.05` | `0.01` |
| **`min_ent`** (entropy decay floor) | `0.005` | `0.005` |
| **`entropy_decay_steps`** | `2000` | `2000` |
| **`vf_coef`** ($c_{value}$) | `0.5` | `0.5` |
| **`lr_start`** (Peak LR) | `1.0e-3` | `3.0e-5` |
| **`lr_end`** (End LR) | `1.0e-5` | `1.0e-6` |
| **`lr_warmup_updates`** | `100` | `100` |
| **`lr_total_updates`** | `5000` | `5000` |
| **`max_grad_norm`** | `0.5` | `1.0` |
| **`opponent`** | `selfplay` | `selfplay` |
| **`heuristic_win_rate`** | `0.6` | `0.6` |
| **`heuristic_window_episodes`** | `100` | `100` |
| **`heuristic_path`** | `versions/.../main.py` | `versions/.../main.py` |
