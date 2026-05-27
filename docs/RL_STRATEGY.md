# RL Strategy For Orbit Wars

This document describes the recommended reinforcement-learning strategy for this project.

The core idea is **hybrid RL**, not raw end-to-end control. Let the neural policy choose among safe, precomputed candidate missions, while deterministic code handles legality, geometry, ship counts, and fallback behavior.

This is the best fit for Orbit Wars because:

- the action space is variable-size and partly continuous;
- legal actions depend on current planet ownership, ships, moving targets, sun collision, and comet state;
- Kaggle inference has a strict per-turn time budget;
- available compute is limited to Kaggle `2 x T4`;
- the current heuristic already encodes useful domain knowledge.

## Target Architecture

```text
observation
  -> GameState parser
  -> candidate mission builder
  -> feature encoder
  -> neural policy ranks candidates
  -> deterministic safety/action decoder
  -> fallback heuristic for unsafe/empty decisions
  -> moves [[planet_id, angle, ships], ...]
```

The policy should not output raw angles. It now has a target head plus a discrete ship-bucket head, but this design needs behavior-cloning supervision and shaped rewards before longer PPO runs.

Instead:

```text
For each owned source planet:
    policy chooses one of K candidate targets or no-op
    policy chooses one discrete ship bucket for the selected target
    deterministic decoder computes:
        - launch angle
        - concrete ship count from the selected bucket
        - path safety
        - final legal move
```

## Action Space

Use one decision row per owned planet.

For every owned source planet, build:

```text
candidate 0: no-op
candidate 1..K: target planet/comet/enemy/friendly reinforce option
```

Recommended first value:

```text
K = 12
```

Candidate target shortlist:

1. nearest neutral planets by travel time;
2. highest ROI neutral planets;
3. weak enemy planets;
4. threatened own planets needing reinforcement;
5. profitable comets with enough remaining life;
6. optional staging/friendly planets for consolidation.

The policy outputs categorical logits over `K + 1` target choices and ship-bucket logits for each target.

Invalid candidates are masked before sampling:

```text
mask = false if:
    source is not owned
    source has too few spare ships
    target is gone/off-board
    path hits sun
    target cannot be captured/reinforced in time
    comet expires too soon
```

## Ship Count Strategy

The current experimental implementation learns a discrete ship bucket rather than raw ship counts.

Concrete bucket counts are generated from mission-specific bases:

```text
0: minimum useful send
1: 0.75x mission base
2: 1.00x mission base
3: 1.25x mission base
4: 1.50x mission base
5: 50% surplus
6: 75% surplus
7: all surplus
```

Use guarded surplus, not raw source ships:

```text
surplus = source ships - source reserve
```

Bucket choices are masked if they overdraw the source, cross the sun, or do not represent a valid mission.

Important current issue: the bucket head is not behavior-cloned yet. Old BC checkpoints initialize target choice but leave bucket choice random. Do not rely on long PPO to discover good ship counts from terminal reward alone.

## Model Inputs

Use scalar features, not images.

### Source Features

Per owned source planet:

```text
x, y
radius
ships
production
spare ships after reserve
is_orbiting
distance_to_center
owned_planet_rank_by_ships
incoming_enemy_ships
incoming_friendly_ships
time_to_nearest_enemy
time_to_nearest_neutral
```

Normalize values:

```text
x, y: / 100
ships: log1p(ships) / log1p(5000)
production: / 5
distances: / 150
times: / 100
```

### Candidate Features

For each candidate:

```text
target type:
    no-op / own / neutral / enemy / comet
target x, y
target ships
target production
target radius
travel_time
distance
ships_needed
ROI estimate
path_hits_sun flag
path_enemy_obstacle flag
future_owner_at_arrival
future_ships_at_arrival
incoming_enemy_to_target
incoming_friendly_to_target
remaining_game_fraction
comet_remaining_life
source_to_target_angle_sin
source_to_target_angle_cos
```

### Global Features

```text
turn / 500
owned planet count
enemy planet count
neutral planet count
owned production
enemy production
owned ships total
enemy ships total
owned fleet ships
enemy fleet ships
military lead ratio
production lead ratio
number of active comets
players alive
```

## Network

Use a small candidate-ranking network.

Recommended first architecture:

```text
source_encoder: MLP(source_dim -> 64 -> 64)
candidate_encoder: MLP(candidate_dim -> 64 -> 64)
global_encoder: MLP(global_dim -> 64 -> 64)

score_head:
    concat(source_emb, candidate_emb, global_emb)
    -> MLP(192 -> 128 -> 1)

value_head:
    mean-pool candidate embeddings
    concat(source_emb, pooled_candidate_emb, global_emb)
    -> MLP(192 -> 128 -> 1)
```

Keep it small:

```text
hidden size: 64 or 128
layers: 2
activation: SiLU or ReLU
```

For Kaggle inference, small MLPs are enough.

## Training Algorithm

Recommended sequence:

1. behavior cloning with both target labels and ship-bucket labels;
2. evaluate BC policy against `random`, `sniper`, and current/best heuristic;
3. add shaped rewards;
4. PPO fine-tune against weak/frozen opponents;
5. PPO self-play with snapshots;
6. league evaluation and model selection.

Do not start directly with pure PPO or pure self-play PPO. Current bucket-action PPO results are weak and indicate an exploration/reward problem, not just insufficient training time.

See `docs/RL_TRAINING_NOTES.md` for the current diagnosis.

## Stage 1: Behavior Cloning

Goal: make the model imitate strong actions, including both target choice and ship amount bucket.

Use direct runner to generate games:

```text
teacher: current main.py
opponents: random, noop, starter, older heuristic snapshots
```

For each teacher/top-player action:

1. reconstruct candidate list for the source planet;
2. identify which candidate target was selected;
3. map real `n_ships` to the closest valid ship bucket for that target;
4. train categorical cross-entropy on target;
5. train categorical cross-entropy on selected target's bucket;
6. train value head from final outcome or shaped return.

Why this matters:

- validates feature encoder;
- validates action decoder;
- gives PPO a reasonable starting policy;
- saves T4 compute.

Target:

```text
model should match teacher target choice >70% when matching candidate exists
model should beat random before PPO
model should have nonzero friendly-send rate before PPO
model should emit bucket distributions close to top-player send amounts
```

## Stage 2: PPO Against Weak Opponents

Train with the direct runner.

Opponent schedule:

```text
20% noop
30% random
20% starter/sniper
30% current heuristic or frozen weak snapshots
```

This teaches expansion and basic combat without immediately collapsing into mirror self-play.

PPO settings to start:

```text
n_envs: as many CPU workers as practical
rollout_steps: 128-512 turns per worker
minibatch_size: 4096-16384 decision rows
ppo_epochs: 2-4
gamma: 0.995
gae_lambda: 0.95
clip_range: 0.1-0.2
entropy_coef: 0.01 initially, decay later
value_coef: 0.5
learning_rate: 1e-4 to 3e-4
max_grad_norm: 0.5
```

Decision rows are per owned planet, so one game step can produce many policy samples.

## Stage 3: Self-Play With Snapshots

Once the model reliably beats weak opponents, introduce snapshot self-play.

Maintain a small opponent pool:

```text
latest model
best model
previous 3-5 checkpoints
current heuristic
random/starter for anti-regression
```

Sampling:

```text
40% latest/best neural snapshots
30% older snapshots
20% heuristic
10% random/starter
```

Freeze snapshots every fixed number of updates or after meaningful evaluation improvement.

Avoid training only against the latest policy; it can overfit to unstable weaknesses.

## Rewards

Use final win/loss as the true objective:

```text
win: +1
loss: -1
draw: 0
```

But for learning speed, add shaped rewards during training only:

```text
delta_total_ships * 0.001
delta_owned_production * 0.02
planet_capture +0.05
planet_loss -0.07
enemy_planet_capture +0.08
fleet_lost_to_sun -0.05
invalid/no-effective-action penalty -0.01
```

Keep shaping small. If shaping dominates win/loss, the model may optimize economy curves while losing games.

Recommended return:

```text
reward = final_result + clipped_shaping
```

Clamp per-step shaping to avoid instability:

```text
step_shaping in [-0.1, 0.1]
```

## Exploration

Use masked categorical sampling.

Do not add random invalid actions.

Entropy should apply only over valid candidates:

```text
logits[~valid_mask] = -inf
```

Exploration knobs:

```text
entropy_coef
temperature during rollout
epsilon no-op/target perturbation
opponent diversity
```

## Inference Strategy

At Kaggle runtime, use deterministic inference:

```text
argmax over valid candidate logits
```

Then safety layer checks:

```text
source still owned
ships still available
path still safe
target still valid
move does not overcommit source
```

If rejected:

```text
try next best candidate
else fallback to heuristic
else no-op
```

This makes the submitted agent robust.

## Compute Plan For 2 x T4

Use GPUs for model updates, CPUs for rollouts.

Recommended setup:

```text
process group A: direct-runner rollout workers
process group B: PPO learner on GPU
queue: rollout batches -> learner
```

If multiprocessing is too complex at first:

1. generate rollout batches on CPU;
2. train PPO updates on GPU;
3. repeat.

Prioritize iteration speed over perfect distributed infrastructure.

Approximate first target:

```text
10M-50M decision rows
```

Because each turn can produce multiple owned-planet decisions, this is much cheaper than 50M full game steps.

## Evaluation

Never trust training reward alone.

Every serious checkpoint:

```text
vs random: 100 games
vs current heuristic: 200 games
vs previous best neural: 200 games
vs mixed snapshot pool: 200 games
```

Track:

```text
win rate
mean final ship score
owned production at turns 25/50/100
sun crash count
invalid action count
mean inference time
```

Promotion rule:

```text
new model becomes candidate best only if it beats current best by >=55% over 200+ games
```

## Submission Packaging

The submitted bot should include:

```text
main.py
src/
configs/
model weights
small inference module
```

Do not include:

```text
training code
direct runner
fast local environment
Numba dependency
Kaggle environment source copy
```

For inference, prefer:

```text
NumPy MLP
small JSON/NPZ weights
deterministic candidate builder
heuristic fallback
```

Avoid PyTorch in final submission unless inference time and package size are confirmed safe.

## What To Build First

### 1. `scripts/cprofile_direct.py`

Profile the current heuristic under the direct runner.

Purpose: reduce teacher rollout cost and identify reusable feature/candidate code.

### 2. `src/candidates.py`

Extract candidate generation from `strategy.py` into reusable deterministic code:

```text
build_candidates(state, source, cfg) -> candidates + mask
decode_candidate(candidate) -> move
```

Both heuristic and RL should use this.

### 3. `rl/features.py`

Encode source/candidate/global features into fixed arrays.

### 4. `rl/bc_train.py`

Behavior cloning from the current heuristic.

### 5. `rl/ppo_train.py`

PPO using direct runner and candidate action masks.

## Main Risks

### Risk: simulator mismatch

Mitigation:

```text
run parity checks constantly
keep official env evaluation in loop
use direct runner only for training speed
```

### Risk: model learns unsafe actions

Mitigation:

```text
valid action masks
deterministic safety decoder
heuristic fallback
```

### Risk: overfitting to weak opponents

Mitigation:

```text
snapshot pool
heuristic opponent
periodic official env evaluation
```

### Risk: inference too slow

Mitigation:

```text
small MLP
top-k candidates
NumPy inference
cache per-turn geometry
fallback early if time budget low
```

## Recommended Immediate Order

1. Profile and optimize current heuristic runtime.
2. Extract reusable candidate builder.
3. Implement feature encoder.
4. Train behavior cloning model.
5. Evaluate BC model against random and heuristic.
6. Start PPO from BC weights.
7. Add self-play snapshot pool.
8. Package smallest successful hybrid model for Kaggle.
