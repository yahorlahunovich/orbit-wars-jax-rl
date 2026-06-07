# Kaggle Heuristics Analysis: Strategy Breakdown & Application

We have cleaned, restructured, and benchmarked the three Kaggle heuristics. This document breaks down the logic of each agent, analyzes their performance, and explains which game conditions they are best suited for.

---

## 1. Agent_1110 (ELO: 1110)
*   **Source**: Adapted and tuned from Vickimar's `orbit-wars-heuristic-1000`.
*   **Language**: Pure Python (~4,800 lines of code).
*   **Core Logic**: 
    - **Multi-Step Forward Simulation**: Uses an internal forward simulator to project planet garrisons and check the outcome of launches $4$ to $20$ turns ahead before executing them.
    - **Speed-Aware Aiming**: Considers the speed-scaling mechanics of fleet sizes; avoids sending small, slow fleets over long distances.
    - **Anti-Second Strategy**: Specifically designed for 4-player games; targets the military/production leader to prevent runaway victories.
*   **Best Suited For**: **4-Player Games and defensive/macro strategies**. Its anti-second logic and forward threat projection make it highly resilient in multi-agent chaotic scenarios.

---

## 2. Producer_Agent_1240 (ELO: 1200)
*   **Source**: Slawek Biel's "The Producer" agent.
*   **Language**: Python + PyTorch helper package (`orbit_lite`).
*   **Core Logic**:
    - **Rule 1: Return-on-Investment (ROI) Focus**: Only launches a fleet if the projected net production gained over the next $H$ turns exceeds the ship count spent.
    - **Rule 2: Idle Ship Reallocation**: If ships are idle and not needed for expansion/threat defense, it consolidates them to friendly planets that are closest to the enemy front line.
*   **Best Suited For**: **Standard 2-Player Games with static front lines**. It maintains strong defensive walls while putting continuous economic pressure on the opponent.

---

## 3. Simplified_Orbit_Wars_Agent_1245 (ELO: 1245)
*   **Source**: Vectorized PyTorch implementation of the "Producer" agent.
*   **Language**: Single-file vectorized PyTorch script (~2,500 lines).
*   **Core Logic**:
    - Implements the exact same planning wave and ROI calculation logic as `Producer_Agent_1240`, but entirely vectorized in PyTorch. 
    - By executing all ray intersections, planet distance lookups, and combat projections in parallel tensor operations, it avoids CPU bottlenecks and operates with slightly tuned parameters.
*   **Best Suited For**: **High-speed, highly-contested 2-Player matches**. It is the most robust and computationally efficient agent in our suite.

---

## 4. Benchmark Performance Comparison (Direct Head-to-Head)
*   **Simplified_Orbit_Wars_Agent_1245 vs. Agent_1110**: **`9 - 1`** in favor of `1245`. The ROI-based planning waves of the Producer agent completely outclass the slower, local-neighborhood heuristic of `1110` in 2-player games.
*   **Simplified_Orbit_Wars_Agent_1245 vs. Producer_Agent_1240**: **`4 - 6`**. This confirmed they are highly matched, running the same core logic with slightly different parameters.

---

## 5. Combined Agent & Strategy Interference Analysis
We built a meta-agent `combined_agent.py` that runs both `Agent_1110` (defensive, simulator-driven) and `Simplified_Orbit_Wars_Agent_1245` (offensive, ROI-driven).
*   **Combiner Logic**: At each step, it runs both heuristics. It prioritizes defensive reinforcement moves from `1110` (to secure threatened planets) and overlays `1245`'s moves for expansion and offensive attacks.
*   **Benchmark Results (Combined vs. 1245)**: **`4 - 11`** in favor of `1245`.

### Strategic Takeaways: Why the Combined Bot Performed Worse
1.  **Garrison Allocation Interference**: `1245` (Producer) depends on moving all idle ships to key front-line planets near the enemy. When we prioritize `1110`'s defensive reinforcement loops, the combined agent locks up ships in internal, defensive planet-to-planet movements. This leaves `1245` with fewer ships to launch offensive waves.
2.  **Conflicting Assumptions**: What `1110` considers a vital defensive reinforcement, `1245` sees as an inefficient lockup of ships that should have gone to the front lines. Mixing rules from conflicting strategic assumptions causes the agents to interfere with each other's optimal ship flow.
3.  **Path to Victory**: Rather than merging distinct action lists at runtime, the best way to leverage both is to use our **Stage Classifier** (from the previous task) to dynamically switch configs or select the dominant agent's strategy globally on a turn-by-turn basis.
