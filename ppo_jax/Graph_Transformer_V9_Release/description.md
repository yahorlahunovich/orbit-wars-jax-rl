# Graph Transformer V9

Graph Transformer V9 is the latest reinforcement learning model for the StarsRL (Orbit Wars) competition.

## Architecture Description
- **Core Trunk:** A standard node transformer (`NodeTransformerLayer`) replacing the all-pairs edge-aware transformer from V8. The policy trunk remains fully trainable during PPO.
- **Edge Features:** Encoded by a lightweight MLP and consumed directly by the policy heads (send, target, bucket heads) rather than being processed by the transformer trunk layers.
- **Value Network:** Fully separate (`EdgeAwareValueNet`) from the policy trunk and receives a lightweight edge summary alongside node features.
- **Policy Heads:** `V9PolicyHead` uses raw edge features and embedded edge representations to output `send` (whether to launch), `target` (where to launch), and `frac` (ship amount bucket) decisions.
- **Future Oracle:** The model relies heavily on a 32-step "future oracle" which provides deterministic future ship amounts on planets based on current fleet trajectories.

## Input/Output Description

### Inputs
The network (`GraphTransformerV9`) receives the following features:
1. **Node Features (`node_features`):** Shape `(N, 21)`. Planet statistics (owner, ships, production, defensive status, etc.).
2. **Future Sight (`future_sight` / Oracle):** Shape `(N, 32)`. The next 32 turns of raw future ship amounts, scaled by `/ 30.0`.
3. **Global Features (`global_features`):** Shape `(8,)`. Game-level statistics (turn progress, total ships, planet ownership percentages).
4. **Edge Features (`edge_features`):** Shape `(N, N, 14)`. Pairwise distance, ETA, blockers, ROI, and relative combat power.

*Total Node Input Dimension into the Node Encoder:* 21 (node) + 32 (future oracle) + 8 (global) = 61.

### Outputs
The network outputs three primary tensors and a baseline value:
1. **Send Logits (`send`):** Shape `(M,)` where M is the number of owned planets. Logits determining the probability of sending a fleet from each owned planet.
2. **Target Logits (`target`):** Shape `(M, N)`. Logits determining the target planet for each sending planet.
3. **Fraction Logits (`frac`):** Shape `(M, N, 3)`. Logits determining the fleet size bucket (e.g., small, medium, large) to send.
4. **Value (`value_head`):** Scalar shape `(1,)`. The baseline value used for PPO training.

## Processing Scripts (Oracle)
Included in this package are the feature extraction and processing scripts required to run the model:
1. `features_jax.py`: Contains the `_extract_planet_future_oracle` implementation and other feature builders (like node and edge extraction) used by the JAX model to prepare inputs.
2. `package_v9_submission.py`: A utility script that compiles the Graph Transformer V9 model, along with the numpy-equivalent Oracle extraction and forward pass logic (`V9_FORWARD`, `V9_BUILD_FEATURES`), into a single `main.py` file for Kaggle submission.

You can inspect the oracle simulation logic inside `features_jax.py` or the generated strings in `package_v9_submission.py`.