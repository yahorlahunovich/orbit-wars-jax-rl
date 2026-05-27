# Bot registry

Official and local results for each saved bot version. Add a new row whenever you submit to Kaggle or freeze a version for comparisons.

**How to benchmark a saved snapshot** (from repo root; paths are relative to this file’s parent directory):

```bash
python scripts/evaluate.py --agent-a versions/baseline_kaggle526/main.py --agent-b random --games 50
```

Swap `--agent-a` / `--agent-b` to pit two snapshots against each other.

---

## Results table

| Bot ID | Name | Date | Kaggle leaderboard | Local benchmark | Strategy principles | Code snapshot |
|--------|------|------|--------------------|-----------------|----------------------|----------------|
| v0 | Baseline heuristic | 2026-05-13 | **526.1** | 50 / 0 / 0 vs `random`, seeds 0–49; mean reward A +1.0 / B −1.0 (`scripts/evaluate.py`) | **Expansion only**—no incoming-fleet defense, no separate comet or attack phase. **Picks sources** as own planets sorted by garrison. **Reserves** `min_defense + 2×production`, sends up to `launch_fraction` of surplus. **Candidates**: up to `max_targets_per_source` nearest non-owned planets (Manhattan). **Scoring**: iterative intercept (`estimate_intercept`) and fleet speed; reject if straight segment source→*current* target hits the sun; require enough ships for neutral garrison or **future** enemy garrison (`production × travel_time`) plus `safety_margin`; score = production weight + short payoff horizon + enemy/comet bonuses − ship and distance cost; skip weak scores; **one fleet per target planet id per turn**. Config: `configs/bot_config.json`. | `versions/baseline_kaggle526/main.py` |
| v1 | ROI + adaptive + defense | 2026-05-14 | **647.7** | *(fill after local run)* e.g. `python scripts/evaluate.py --agent-a main.py --agent-b random --games 50` | **Defense first** (pooled, ETA-aware reinforcements vs aggregated ray threats). **Expansion**: greedy **ROI** pick each launch (payoff per ship, then composite score, then shorter travel); **patient** vs enemy-owned, lighter margin vs neutrals; **simulate in-flight arrivals** at target; paths must clear **sun + enemy planets**; **adaptive** `travel_time_max`, `payoff_min_turns`, buffer vs military lead vs strongest opponent. Comet travel cap + moderated comet ROI bump. See `configs/bot_config.json` (`adaptive_*`). | *(freeze to `versions/kaggle647/` when you want a fixed benchmark copy)* |
| v2 | Current heuristic | 2026-05-19 | **700** | Direct runner smoke/eval used during RL work; rerun with `conda run -n ml python scripts/bench_direct.py --agent-a versions/kaggle700_current_heuristic/main.py --agent-b noop --games 3 --episode-steps 200 --kaggle-env-root analysis/fast_kaggle_env` for speed. | Current strongest heuristic snapshot. Includes smooth reserve ramp, adaptive lead/behind tuning, incoming-fleet bookkeeping, defense-first reinforcements, target arrival simulation, path safety against sun/enemy planets, greedy ROI expansion, comet travel cap, and intercept aiming against rotating planets. | `versions/kaggle700_current_heuristic/main.py` |

---

## Column notes

- **Kaggle leaderboard**: public competition score at the time you recorded it (same units as the leaderboard UI).
- **Local benchmark**: quick, reproducible check in this repo (opponent, seeds, script); extend with more seeds or a second snapshot when comparing versions.
- **Strategy principles**: short text summary of how the bot decides; keep the row updated when behavior changes meaningfully.
- **Code snapshot**: self-contained tree (`main.py`, `src/`, `configs/`) so `evaluate.py` loads that version’s `src`, not the evolving repo root. See `versions/README.md` for how to add new snapshots.
