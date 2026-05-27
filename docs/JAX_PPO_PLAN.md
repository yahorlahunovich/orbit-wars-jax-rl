# JAX PPO Training Plan

This document turns `atchitecture.md` into a concrete, step-by-step implementation plan.
It targets the JAX stack only (`rl_training_jax/`). The legacy PyTorch path stays as a
reference / opponent.

## Design summary (derived from `atchitecture.md`)

| Item | Decision |
|------|----------|
| Framework | JAX + Flax + Optax |
| Env | `rl_training_jax/src/orbit_wars/` (already JAX-native, vectorized) |
| Rollout parallelism | `jax.vmap(step_jit)` over many envs in one big JIT |
| Rollout steps per env | 32 |
| Effective batch | ≥ 2048 decision rows per minibatch |
| Observations | Rich, decomposed scalar features per planet/fleet/comet (target up to ~1000 features in aggregate, but each entity has 16–32 scalars). Goal: explained_variance > 0.9. |
| Architecture | Small Transformer encoder over (planets ⊕ fleets ⊕ comets); heads for value, target, fleet size bucket |
| Action | Per owned planet: pick **a planet** (own/enemy/neutral/comet) as target + ship bucket. Angle computed geometrically. |
| Reward | Pure terminal +1 / 0 / -1. No shaping. |
| Discount | γ very close to 1 (0.9999 or 1.0) |
| Algorithm | PPO with cosine-decayed LR |
| Training mode | Self-play |
| Logged metrics | clip_fraction, approx_kl, explained_variance, steps/sec |
| Throughput target | 10,000 env-steps/sec (achieved on CPU with vmap≥16; will be much higher on Kaggle GPU) |

## Where this differs from the legacy PyTorch pipeline

| Concern | Legacy (`rl_training/`) | New JAX plan |
|---------|-------------------------|--------------|
| Action set | K nearest candidate planets per source | **All planets** are valid targets (we let the network learn) |
| Output | candidate_logits + ship_bucket_logits | target_planet_logits + ship_bucket_logits + value, via Transformer pooling |
| Reward | terminal + shaping (production/ships/captures) | **pure terminal** ±1 |
| Opponent | scripted heuristic or random | **self-play** (current policy plays both sides) |
| Features | hand-engineered per source candidate | hand-engineered **per entity**, fed into a Transformer |
| Throughput | Python env, ~100–300 sps | JIT vmap env, **≥10k env-sps** |

The legacy stack stays around for benchmarking and as a known-good opponent during eval, but the new training run does not touch it.

## End-to-end pipeline

```
reset (vmap)                                    # B envs
    └─ pad to (B, MAX_PLANETS, F_planet) etc.
        │
        ▼
encode_obs_jax                                  # all JAX, no numpy bridge
        │
        ▼
PlanetTransformerPolicy(params, obs)            # Flax, jitted
        ├─ planet_logits  (B, MAX_PLANETS, MAX_PLANETS)   # per-source target picker
        ├─ bucket_logits  (B, MAX_PLANETS, BUCKETS)
        └─ value          (B,)
        │
        ▼
sample masked actions, geometry-compute angles
        │
        ▼
batched_step (vmap step_jit)                    # one JIT, B envs
        │
        ▼
GAE on flat decision rows                       # JAX
        │
        ▼
PPO minibatch update                            # JAX
```

---

## Implementation plan (10 phases)

> Each phase is small enough to validate independently. Do **not** skip the
> validation step at the end of any phase — they are explicit guardrails.

### Phase 1 — JAX-native feature encoder

**Goal:** replace the numpy `encode_turn` bridge with a pure-JAX `encode_obs` that
operates directly on `OrbitWarsState`. No more `state_to_observation_dict` /
numpy round-trips inside the rollout.

Files to add:

- `rl_training_jax/src/orbit_wars/features_jax.py`

Features per planet (≈22 scalars):

```
owner_is_me, owner_is_enemy, owner_is_neutral
ships_log,                                     # log1p(ships) / log1p(5000)
production / 5
radius / 10
x / 100, y / 100
distance_to_center / 50
is_orbiting, is_comet
incoming_enemy_ships, incoming_friendly_ships  # summed over fleets
time_to_nearest_enemy_fleet
distance_to_my_capital
ROI = production / (ships + 1)
reachable_without_sun                          # geometry helper
remaining_path_steps (comet only, else 0)
planet_value_rank                              # 0..1 ordinal among all planets
ships_log_relative_to_max
nearest_enemy_distance / 50
nearest_friendly_distance / 50
```

Features per fleet (≈8 scalars):

```
owner_is_me, owner_is_enemy
ships_log, fleet_speed
travel_remaining_time
from_planet_id_onehot summary (skip — Transformer attends)
heading_sin, heading_cos
```

Global features (≈16 scalars):

```
turn / 500
remaining_game_fraction
my_planet_count, enemy_planet_count, neutral_planet_count   # / MAX_PLANETS
my_production, enemy_production                              # normalized
my_ships_total, enemy_ships_total                            # log
production_lead, ship_lead
active_comet_count
players_alive
my_fleet_count, enemy_fleet_count
incoming_to_me_total, outgoing_from_me_total
```

All masks pre-computed from `planets[:, 7]` and ownership.

**Validation:**
- Numerical: feed a real `OrbitWarsState`, check shapes
  `(MAX_PLANETS, F_planet), (MAX_FLEETS, F_fleet), (F_global,)`.
- Property: re-encode after `step_jit` shows monotone `turn / 500` and updates ownership counts.
- `vmap(encode_obs)` jits and runs without recompile.

### Phase 2 — Transformer policy

**Goal:** small Flax Transformer that consumes per-entity features and emits
`(target_logits per source, bucket_logits per source, value)`.

Files to add/modify:

- Replace `rl_training_jax/src/policy.py` with a transformer model.

Architecture (start small):

```python
class PlanetTransformerPolicy(nn.Module):
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    bucket_count: int = 8
    ff_mult: int = 4
```

Forward:

```
planet_tokens   = Dense(d_model)(planet_features)            # (B, P, d)
fleet_tokens    = Dense(d_model)(fleet_features)             # (B, F, d)
type_emb        = type-of-token learned embedding
global_token    = Dense(d_model)(global_features)[:, None, :] # (B, 1, d)

tokens = concat([global_token, planet_tokens, fleet_tokens], axis=1)
mask   = concat([all_one, planet_mask, fleet_mask], axis=1)

for _ in range(num_layers):
    tokens = TransformerBlock(d_model, num_heads, ff_mult)(tokens, mask)

planet_h = tokens[:, 1 : 1 + P, :]              # per-planet hidden
global_h = tokens[:, 0, :]                       # CLS-like

# Target head: per source planet, score each candidate planet (P -> P)
# (B, P_src, d) x (B, P_tgt, d) -> (B, P_src, P_tgt)
target_logits = einsum("bsd,btd->bst", planet_h, planet_h) / sqrt(d_model)
# Bucket head: per source planet, score each ship bucket
bucket_logits = MLP(d_model -> bucket_count)(planet_h)        # (B, P, BUCKETS)

# Value head: from global token
value = MLP(d_model -> 1)(global_h).squeeze(-1)
```

Masks:
- A source planet decision is valid only if it is owned by the active player.
- A target planet is valid only if the planet slot is active.
- Bucket is valid only if it yields ships > 0 given source ships.

**Validation:**
- Pure shape test (`tests/test_transformer.py`).
- Smoke forward+backward in `jit`.
- Param count ≤ 1.5M (small enough for Kaggle inference and fast updates).

### Phase 3 — Geometry decoder (angle from chosen target)

**Goal:** deterministic JAX function that turns `(source_planet, target_planet, ship_count)`
into a legal `(from_id, angle, ships)` move row.

Files to add:

- `rl_training_jax/src/orbit_wars/decode.py`

Logic:

1. Use `estimate_intercept` (port from `rl_training/src/geometry.py`) so we lead
   moving / orbiting / comet targets.
2. If predicted path crosses the sun, mark the move invalid (mask the target
   in the policy step before sampling).
3. Compute concrete ship count from the bucket index:

```
bucket 0: max(1, 0.1 * source_ships)
bucket 1: max(1, 0.25 * source_ships)
bucket 2: max(1, 0.5  * source_ships)
bucket 3: max(1, 0.75 * source_ships)
bucket 4:        source_ships        # all-in
bucket 5: target_ships + 1            # exact capture
bucket 6: target_ships + 0.5 * source_ships
bucket 7: minimum useful send (4)
```

Buckets are masked if they exceed source ships or yield 0.

**Validation:**
- Round-trip test: emit a move, run `step`, check the resulting fleet matches
  the source/angle within geometric tolerance.

### Phase 4 — Action sampling + packing

**Goal:** sample `(target_idx, bucket_idx)` per owned planet, then pack into the
`(MAX_MOVES_PER_PLAYER, 3)` action tensor for `step_jit`.

Files to add:

- `rl_training_jax/src/orbit_wars/rollout.py` (sample + pack)

Highlights:

- Mask invalid sources, targets, and buckets before sampling. Required for entropy to live on valid moves only.
- One `jax.random.categorical` per source planet for target, one for bucket.
- Each owned planet produces at most one move per turn. Source planets that are not owned emit `mask=0` rows.

**Validation:**
- `rollout(params, state, rng) -> (state', actions, log_probs, value, entropy)` is fully `jit`'d.
- A pure-random init policy generates a legal mix of `noop` / valid moves over 100 steps without crashes.

### Phase 5 — Vectorized self-play rollout

**Goal:** collect `(B, T)` decision rows in one giant JIT scan.

Files to add:

- `rl_training_jax/src/train_ppo.py`
- `rl_training_jax/configs/transformer_selfplay.yaml`

Design:

```python
def rollout_step(carry, _):
    states, rng = carry
    rng, kp0, kp1 = jax.random.split(rng, 3)
    actions_p0, lp0, val0, ent0, target0, bucket0, mask0 = sample(params, states, player=0, kp0)
    actions_p1, lp1, val1, ent1, target1, bucket1, mask1 = sample(params, states, player=1, kp1)
    states = batched_step(states, actions_p0, actions_p1, mask0, mask1)
    # record per-decision rows for the *learner* (player 0 by default,
    # alternate to share data)
    return (states, rng), (states, lp0, val0, target0, bucket0, mask0, states.rewards, states.done)

(states, rng), traj = jax.lax.scan(rollout_step, (states, rng), None, length=cfg.rollout_steps)
```

Key choices:

- **Self-play:** both sides use the same `params`. Periodically alternate which
  player's rows are used as learner targets so both perspectives are seen.
- **Done handling:** when `done`, reset that env (deterministic seed bump).
  `jax.lax.cond` or `jnp.where` over the state pytree.
- **Comet spawn:** keep numpy spawn for now (5 events per game, negligible).
  Trigger only on host every `rollout_steps * num_envs` between scans. Document
  that scan length should be ≤ 50 to stay between spawn turns, OR pre-spawn all
  5 comet groups at reset (see Phase 9).
- **Action storage:** flatten `(B, T, MAX_PLANETS)` decision rows into `(N, ...)` for PPO.
  Apply `mask` to filter invalid rows.

Per-update sizing:

```
num_envs           = 64        # tunable (vmap width)
rollout_steps      = 32        # from architecture.md
decision_rows ≈ owned_planets_per_env * num_envs * rollout_steps
              ≈ 6 * 64 * 32 ≈ 12k rows  (well above the 2048 minibatch target)
```

### Phase 6 — GAE + PPO update

**Goal:** GAE(γ ≈ 1.0, λ = 0.95), PPO clip, value loss, entropy bonus, cosine LR.

Files to modify:

- `rl_training_jax/src/ppo.py`

Reward / advantage rules (per `atchitecture.md` §12):

- Terminal reward only: +1 win, -1 loss, 0 draw / ongoing.
- `gamma = 0.9999` (effectively 1.0 over 500 steps).
- `gae_lambda = 0.95`.
- Advantages normalized per batch (mean 0 / std 1).

PPO config (defaults from architecture):

```yaml
gamma: 0.9999
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.01      # decay later
vf_coef: 0.5
epochs: 3
minibatch_size: 2048
lr_schedule: cosine
lr_start: 1.0e-3
lr_end:   1.0e-5
max_grad_norm: 0.5
```

Required logged metrics every update:

```
- update / sec
- env-steps / sec        (target: >= 10k)
- explained_variance     (target: > 0.9 after warm-up)
- approx_kl              (target: stable, ~0.005–0.02)
- clip_fraction          (target: 0.05–0.3, NOT zero, NOT >0.5)
- entropy                (declining slowly)
- episode_return_mean
- self_play_winrate      (always ~0.5 by construction; sanity check)
```

### Phase 7 — Smoke run (single GPU / CPU)

**Goal:** prove the whole loop runs end to end without parity errors and that
explained_variance starts climbing.

Config: `configs/smoke_transformer.yaml`

```yaml
seed: 0
run_name: jax_smoke_transformer
env:
  episode_steps: 200
  num_envs: 16
ppo:
  total_updates: 20
  rollout_steps: 32
  minibatch_size: 1024
  epochs: 2
model:
  d_model: 64
  num_layers: 2
  num_heads: 4
```

Pass criteria:

- runs without numerical NaN
- `clip_fraction > 0` and `< 0.5` by update 10
- `env_sps > 5000`
- `approx_kl` finite and bounded
- `explained_variance` improves over the 20 updates

### Phase 8 — Eval vs heuristic & previous-best

**Goal:** verify the new agent has signal vs the existing `versions/kaggle700_current_heuristic/` opponent.

Files to add:

- `rl_training_jax/scripts/eval_jax_vs_heuristic.py`

Plan:

- Load latest checkpoint.
- Play 100 games against `versions/kaggle700_current_heuristic/main.py` via the
  direct runner (`scripts/bench_direct.py` patterns).
- Report win-rate, mean score, sun losses, illegal-move count.

### Phase 9 — Pre-baked comets (full GPU vmap path)

**Goal:** eliminate the last Python step inside `step()` so a full episode is a
single JIT scan (no host round-trips).

Approach:

- At reset, pre-generate **all 5 comet groups** for the entire episode (the
  RNG is deterministic from `episode_seed`). Mark each group inactive until
  its spawn step. The JIT path activates them via a mask check on `state.step`.
- Adds a new state field `comet_spawn_step: (MAX_COMET_GROUPS,) int32`.
- After this, `batched_step` can be wrapped in `jax.lax.scan` of length
  `episode_steps` for fully on-device rollouts (gigantic GPU speedups).

This is an optional optimization. Do it only after Phase 7 proves the loop is sound.

### Phase 10 — Kaggle GPU run + submission export

**Goal:** longer training on Kaggle T4, export weights to a small inference bot.

Plan:

- Build a Kaggle notebook from `scripts/build_kaggle_jax_notebook.py` (already exists; update with the new training entry point).
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`.
- Target ~5–10k updates (≈10–50M env-steps).
- Export `params` as a `.npz` blob + a NumPy inference path for the submitted `main.py`. JAX CPU wheels are small enough for the Kaggle submission tarball; fallback is NumPy MLP.

---

## Risk register

| Risk | Symptom | Mitigation |
|------|---------|------------|
| Reward signal too sparse with γ≈1, ±1 only | `explained_variance` stays ≈0, advantages noisy | Allow optional shaping only as a debug toggle (`shaping=0` is default). Train longer; bigger num_envs. |
| Self-play collapse to a fixed strategy | win-rate oscillates around 0.5 but no skill gain vs heuristic | Maintain a snapshot pool (refresh every N updates), sample 30% of opponents from the pool. |
| Transformer too big for Kaggle inference | submission timeout | Cap at 3 layers, d_model 96, num_heads 4. Export to NumPy. |
| Mask bugs (sampling invalid actions) | clip_fraction explodes, `approx_kl > 0.1` | Unit-test sampler: all sampled targets have `mask = True`. Asserts in rollout. |
| `done` resets break gradients | spurious huge advantages | Test: feed an artificial done at step T, ensure GAE bootstrapping zeros out future returns correctly. |
| Comet spawn host round-trip ruins JIT vmap | low `env_sps` despite big batch | Phase 9 (pre-baked comets). |

---

## Concrete file layout after the plan lands

```
rl_training_jax/
  src/
    orbit_wars/
      ...                                # already vectorized (this PR)
      features_jax.py        # Phase 1 — pure JAX encoder
      decode.py              # Phase 3 — angle from target, bucket -> ships
      rollout.py             # Phase 4 — sample + pack
    policy.py                # Phase 2 — Transformer
    ppo.py                   # Phase 6 — GAE + PPO loss
    train_ppo.py             # Phase 5 — scan-based self-play loop
  configs/
    smoke_transformer.yaml   # Phase 7
    transformer_selfplay.yaml
  scripts/
    bench_speed.py           # already updated this PR with vmap bench
    eval_jax_vs_heuristic.py # Phase 8
    build_kaggle_jax_notebook.py
  tests/
    test_features_jax.py     # Phase 1
    test_transformer.py      # Phase 2
    test_decode.py           # Phase 3
    test_sample_mask.py      # Phase 4
    test_parity.py           # (this PR — full episode + vmap parity)
    test_geometry.py
    test_policy.py
```

---

## Order of execution (recommended)

1. **Phase 1** (features_jax.py) — small, well-tested.
2. **Phase 2** (transformer policy) — replace the MLP, validate forward + grad.
3. **Phase 3** (decode.py) — pure functional, easy to unit-test.
4. **Phase 4** (rollout.py) — sample + pack with mask validation.
5. **Phase 6** (PPO update) — drop in cosine LR, GAE with γ=0.9999.
6. **Phase 5** (train_ppo.py) — wire everything into a scan loop.
7. **Phase 7** (smoke run) — 20 updates, sanity gates.
8. **Phase 8** (eval vs heuristic) — first external win-rate datapoint.
9. **Phase 9** (pre-baked comets) — only if env_sps plateaus.
10. **Phase 10** (Kaggle long-run + submission).

Before starting Phase 1 in code, confirm:

- Architecture decisions (especially feature list and Transformer size).
- Self-play data sharing (both players' rows used as training data, or only learner side).
- Whether comet pre-baking (Phase 9) should be done earlier.
