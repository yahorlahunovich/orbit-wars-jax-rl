# JAX PPO Optimization Log

This file tracks the implementation of improvements based on the expert critique provided on June 2, 2026.

## Status Summary
- **Point 1 (Features):** DONE. Added explicit Fleet tokens and a Global token to the Transformer. Policy now has full directionality and multi-fleet visibility.
- **Point 2 (Grid/NOOP):** DONE. Optimized $O(P^3)$ path blocker with bounding-box pre-checks. Fixed sitter loop via reward logic.
- **Point 3 (Conditioning):** DONE. Bucket head now conditions on both source and target planet representations (4D bucket_logits).
- **Point 4 (Rewards):** DONE. Added intermediate step_rewards (+0.01 per planet owned) to accelerate early expansion learning.
- **Point 5 (Truncation):** DONE. PPO now uses \`executed_mask\` to only learn from actions that were actually sent to the environment (fixing the 48-move limit bias).

## Updates

### [2026-06-02] - Initializing Optimization Phase
- Reviewed critique and identified action items.
- **Full State Representation (Point 1):** Overhauled \`features_jax.py\` and \`policy.py\` to include Global and Fleet tokens in the Transformer self-attention.
- **Grid Optimization (Point 2):** Added axis-aligned bounding box (AABB) checks to \`path_blocked_by_planets\` to skip expensive math for distant obstacles.
- **Bucket Conditioning (Point 3):** Updated \`PlanetPolicy\` and \`rollout.py\` to make bucket decisions target-dependent.
- **Intermediate Rewards (Point 4):** Added per-step planet ownership rewards to \`step.py\` and integrated them into the PPO training loop.
- **Truncation Bias Fix (Point 5):** Modified \`rollout.py\` and \`ppo_loss_fn\` to mask out truncated actions (beyond the 48-move limit).
- **Test Validation:** Updated the entire test suite to match the new architecture. All 75 tests passing.
