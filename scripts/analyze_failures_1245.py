import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic")
from src.game import parse_state, Planet, Fleet, GameState
from direct_runner import run_direct_from_names

def analyze_match(seed, agent_0_path, agent_1_path, agent_0_label, agent_1_label):
    try:
        steps, elapsed = run_direct_from_names(
            [agent_0_path, agent_1_path],
            root=Path("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2"),
            seed=seed,
            episode_steps=500,
            keep_steps=True
        )
    except Exception as e:
        print(f"Error running seed {seed}: {e}")
        return None
        
    final_step = steps[-1]
    # Check rewards
    r0 = final_step[0].reward
    r1 = final_step[1].reward
    
    # Identify winner
    if r0 > r1:
        winner = agent_0_label
    elif r1 > r0:
        winner = agent_1_label
    else:
        winner = "Draw"
        
    # We want to analyze this match in detail
    T = len(steps)
    history = []
    
    for t in range(T):
        # Parse state from perspective of player 0 (doesn't matter, it has all planets)
        obs_p0 = steps[t][0]
        obs = getattr(obs_p0, 'observation', None)
        if obs is None:
            continue
            
        # Convert SimpleNamespace to dict if needed
        if not isinstance(obs, dict):
            obs_dict = {
                "planets": getattr(obs, "planets", []),
                "fleets": getattr(obs, "fleets", []),
                "player": 0,
                "angular_velocity": getattr(obs, "angular_velocity", 0.0),
                "initial_planets": getattr(obs, "initial_planets", []),
                "comet_planet_ids": getattr(obs, "comet_planet_ids", []),
                "comets": getattr(obs, "comets", []),
                "step": getattr(obs, "step", getattr(obs, "turn", 0)),
                "remainingOverageTime": getattr(obs, "remainingOverageTime", 60.0),
            }
        else:
            obs_dict = dict(obs)
            obs_dict['player'] = 0
            
        state = parse_state(obs_dict)
        
        # Calculate ship counts
        ships_p0 = sum(p.ships for p in state.planets if p.owner == 0) + sum(f.ships for f in state.fleets if f.owner == 0)
        ships_p1 = sum(p.ships for p in state.planets if p.owner == 1) + sum(f.ships for f in state.fleets if f.owner == 1)
        
        planets_p0 = sum(1 for p in state.planets if p.owner == 0)
        planets_p1 = sum(1 for p in state.planets if p.owner == 1)
        
        prod_p0 = sum(p.production for p in state.planets if p.owner == 0)
        prod_p1 = sum(p.production for p in state.planets if p.owner == 1)
        
        history.append({
            'step': t,
            'ships_p0': ships_p0,
            'ships_p1': ships_p1,
            'planets_p0': planets_p0,
            'planets_p1': planets_p1,
            'prod_p0': prod_p0,
            'prod_p1': prod_p1,
        })
        
    # Find pivot turn: when did the loser fall behind and never recover?
    # Let's say we analyze from perspective of 1245
    p_1245 = 0 if agent_0_label == "1245" else 1
    p_opp = 1 - p_1245
    
    pivot_turn = -1
    for t in range(T - 1):
        h = history[t]
        ships_1245 = h[f'ships_p{p_1245}']
        ships_opp = h[f'ships_p{p_opp}']
        
        # If 1245 falls behind by more than 15% of total ships, and remains behind till end
        if ships_1245 < ships_opp * 0.85:
            # Check if it remains behind
            remains_behind = True
            for future_t in range(t, T):
                fut_h = history[future_t]
                if fut_h[f'ships_p{p_1245}'] >= fut_h[f'ships_p{p_opp}']:
                    remains_behind = False
                    break
            if remains_behind:
                pivot_turn = t
                break
                
    # Gather info on planet captures around pivot_turn
    capture_events = []
    if pivot_turn != -1:
        # Look 30 turns before and 10 turns after pivot_turn
        start_t = max(0, pivot_turn - 30)
        end_t = min(T - 1, pivot_turn + 10)
        
        # Track planet ownership
        planet_owners_prev = {}
        for t in range(start_t, end_t + 1):
            obs_p0 = steps[t][0]
            obs = getattr(obs_p0, 'observation', None)
            if obs is None:
                continue
            planets = getattr(obs, 'planets', [])
            current_owners = {int(p[0]): int(p[1]) for p in planets}
            
            if planet_owners_prev:
                for pid, owner in current_owners.items():
                    prev_owner = planet_owners_prev.get(pid, -1)
                    if prev_owner != owner:
                        # Owner changed!
                        capture_events.append({
                            'step': t,
                            'planet_id': pid,
                            'from_owner': prev_owner,
                            'to_owner': owner,
                            'production': float(planets[pid][6]) if pid < len(planets) else 0.0
                        })
            planet_owners_prev = current_owners
            
    return {
        'seed': seed,
        'agent_0': agent_0_label,
        'agent_1': agent_1_label,
        'r0': r0,
        'r1': r1,
        'winner': winner,
        'pivot_turn': pivot_turn,
        'capture_events': capture_events,
        'history': history,
        'final_ships_1245': history[-1][f'ships_p{p_1245}'] if history else 0,
        'final_ships_opp': history[-1][f'ships_p{p_opp}'] if history else 0,
    }

def main():
    agent_1245_path = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/heuristics/Simplified_Orbit_Wars_Agent_1245/main.py"
    agent_700_path = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic/main.py"
    
    seeds = list(range(100, 150)) # 50 seeds
    results = []
    
    print("Running 100 matches (50 seeds, both positions) in parallel...")
    
    # Prepare task arguments
    tasks = []
    for seed in seeds:
        # 1245 as player 0
        tasks.append((seed, agent_1245_path, agent_700_path, "1245", "kaggle700"))
        # 1245 as player 1
        tasks.append((seed, agent_700_path, agent_1245_path, "kaggle700", "1245"))
        
    num_workers = min(mp.cpu_count(), 16)
    with mp.Pool(num_workers) as pool:
        results_nested = pool.starmap(analyze_match, tasks)
        
    results = [r for r in results_nested if r is not None]
    
    # Summarize win rate
    wins_1245 = sum(1 for r in results if r['winner'] == '1245')
    wins_700 = sum(1 for r in results if r['winner'] == 'kaggle700')
    draws = sum(1 for r in results if r['winner'] == 'Draw')
    total = len(results)
    
    print("\n=== Match Summary ===")
    print(f"Total games: {total}")
    print(f"Wins 1245: {wins_1245} ({wins_1245/total*100:.2f}%)")
    print(f"Wins kaggle700: {wins_700} ({wins_700/total*100:.2f}%)")
    print(f"Draws: {draws} ({draws/total*100:.2f}%)")
    
    # Analyze failures
    failures = [r for r in results if r['winner'] == 'kaggle700']
    print(f"\nAnalyzing {len(failures)} failures of 1245...")
    
    pivot_turns = []
    capture_reasons = []
    
    for f in failures:
        pivot = f['pivot_turn']
        if pivot != -1:
            pivot_turns.append(pivot)
            
        # Look at captures around pivot turn
        caps = f['capture_events']
        p_1245 = 0 if f['agent_0'] == '1245' else 1
        p_opp = 1 - p_1245
        
        # Did opponent capture one of our planets or a neutral?
        for cap in caps:
            if cap['to_owner'] == p_opp:
                if cap['from_owner'] == p_1245:
                    capture_reasons.append(f"Opponent captured 1245 owned planet (ID: {cap['planet_id']}, Prod: {cap['production']}) at turn {cap['step']}")
                elif cap['from_owner'] == -1:
                    capture_reasons.append(f"Opponent captured neutral planet (ID: {cap['planet_id']}, Prod: {cap['production']}) at turn {cap['step']}")
                    
    if pivot_turns:
        print(f"Average Pivot Turn (when 1245 falls behind decisively): {np.mean(pivot_turns):.2f}")
    else:
        print("No decisive pivot turns found.")
        
    print("\nSample Capture Events around failure points:")
    for r in capture_reasons[:15]:
        print(f"  - {r}")
        
    # Write a summary analysis artifact
    artifact_dir = "/home/yahor/.gemini/antigravity-cli/brain/fa8225ec-c931-461c-a04e-13f5bbf39638"
    summary_path = os.path.join(artifact_dir, "failures_analysis_1245.json")
    
    with open(summary_path, "w") as f_out:
        json.dump({
            'total_games': total,
            'wins_1245': wins_1245,
            'wins_700': wins_700,
            'draws': draws,
            'pivot_turns': pivot_turns,
            'capture_reasons': capture_reasons[:50]
        }, f_out, indent=2)
        
    print(f"\nDetailed analysis JSON saved to {summary_path}")

if __name__ == "__main__":
    main()
