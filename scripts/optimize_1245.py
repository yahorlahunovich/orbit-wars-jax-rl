import os
import sys
import json
import math
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp

# Ensure workspace root and direct_runner are in path
sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/scripts")
sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic")

from direct_runner import run_direct, resolve_agent, load_agent_from_file
from src.game import parse_state

# Helper to load and monkey-patch 1245 agent
def make_patched_agent(path, config_dict, name):
    # Load module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    
    # Create customized config
    import dataclasses
    base_config = module.ProducerLiteConfig()
    custom_config = base_config
    for k, v in config_dict.items():
        custom_config = dataclasses.replace(custom_config, **{k: v})
        
    # Monkey-patch config_for and CONFIG_4P
    module.CONFIG_4P = custom_config
    module._config_for = lambda pc: custom_config
    
    # Return the agent function
    return module.agent

# Helper to parse steps and calculate metrics
def evaluate_metrics(steps, p):
    T = len(steps)
    records = []
    
    # Pre-parse all states
    states = []
    actions = []
    for t in range(T):
        step_p = steps[t][p]
        if isinstance(step_p, dict):
            obs = step_p.get('observation', {})
            act = step_p.get('action', [])
        else:
            obs = getattr(step_p, 'observation', None)
            act = getattr(step_p, 'action', [])
        actions.append(act)
        
        if isinstance(obs, dict):
            obs_copy = dict(obs)
        else:
            obs_copy = {
                "planets": getattr(obs, "planets", []),
                "fleets": getattr(obs, "fleets", []),
                "player": p,
                "angular_velocity": getattr(obs, "angular_velocity", 0.0),
                "initial_planets": getattr(obs, "initial_planets", []),
                "comet_planet_ids": getattr(obs, "comet_planet_ids", []),
                "comets": getattr(obs, "comets", []),
                "step": getattr(obs, "step", getattr(obs, "turn", 0)),
                "remainingOverageTime": getattr(obs, "remainingOverageTime", 60.0),
            }
        obs_copy['player'] = p
        state = parse_state(obs_copy)
        states.append(state)
        
    ships_launched = 0
    ships_expansion = 0
    ships_attack = 0
    ships_reinforce = 0
    reserve_ratios = []
    
    for t in range(1, T):
        state = states[t]
        my_planets_list = [planet for planet in state.planets if planet.owner == p]
        if not my_planets_list:
            continue
            
        my_ships_planets = sum(planet.ships for planet in my_planets_list)
        my_ships_fleets = sum(f.ships for f in state.fleets if f.owner == p)
        my_ships_total = my_ships_planets + my_ships_fleets
        
        if t >= 100 and t <= 400:
            reserve_ratios.append(my_ships_planets / max(1.0, my_ships_total))
            
        action = actions[t]
        if action and isinstance(action, list):
            planet_owners = {pl.id: pl.owner for pl in state.planets}
            for move in action:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                ships = int(ships)
                
                # Check target
                # A simple estimation: if we are launching in angle direction, what is target owner?
                # We will approximate based on direct_runner geometry helpers if imported
                # For simplicity here, let's just use a simple dot product or closest planet
                # to get target owner
                from_planet = next((pl for pl in state.planets if pl.id == from_id), None)
                if not from_planet:
                    continue
                
                # Find closest planet along the launch ray
                best_dist = float('inf')
                best_owner = -1
                ux = math.cos(angle)
                uy = math.sin(angle)
                
                for pl in state.planets:
                    if pl.id == from_id:
                        continue
                    dx = pl.x - from_planet.x
                    dy = pl.y - from_planet.y
                    # Distance projection
                    proj = dx * ux + dy * uy
                    if proj <= 0:
                        continue
                    perp_dist = abs(dx * (-uy) + dy * ux)
                    if perp_dist <= pl.radius + 2.0:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < best_dist:
                            best_dist = dist
                            best_owner = pl.owner
                            
                if best_owner == -1:
                    ships_expansion += ships
                elif best_owner == p:
                    ships_reinforce += ships
                else:
                    ships_attack += ships
                    
                ships_launched += ships
                
    avg_reserve = np.mean(reserve_ratios) if reserve_ratios else 0.8
    total_launches = max(1, ships_launched)
    
    return {
        'reserve_ratio': avg_reserve,
        'expansion_pct': ships_expansion / total_launches,
        'attack_pct': ships_attack / total_launches,
        'reinforce_pct': ships_reinforce / total_launches,
    }

def run_single_match(agent_a_fn, agent_b_fn, seed, label_a, label_b):
    try:
        steps, elapsed = run_direct(
            [agent_a_fn, agent_b_fn],
            seed=seed,
            episode_steps=500,
            keep_steps=True
        )
    except Exception as e:
        print(f"Error running match seed {seed}: {e}")
        return None
        
    final_step = steps[-1]
    r0 = final_step[0].reward
    r1 = final_step[1].reward
    
    winner = 0 if r0 > r1 else (1 if r1 > r0 else -1)
    
    # Parse metrics for both
    metrics_a = evaluate_metrics(steps, 0)
    metrics_b = evaluate_metrics(steps, 1)
    
    return {
        'seed': seed,
        'winner': winner,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b,
    }

def run_evaluation(config_dict, name, num_games=40):
    agent_path = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/heuristics/Simplified_Orbit_Wars_Agent_1245/main.py"
    
    # Load baseline agent (no patch)
    baseline_agent = load_agent_from_file(Path(agent_path), Path("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2"))
    
    # Load custom patched agent
    custom_agent = make_patched_agent(agent_path, config_dict, name)
    
    wins_custom = 0
    wins_baseline = 0
    draws = 0
    
    reserves = []
    attacks = []
    reinforces = []
    
    # Run games in parallel
    pool_args = []
    for seed in range(500, 500 + num_games):
        if seed % 2 == 0:
            # Custom is player 0, Baseline is player 1
            pool_args.append((custom_agent, baseline_agent, seed, "custom", "baseline"))
        else:
            # Baseline is player 0, Custom is player 1
            pool_args.append((baseline_agent, custom_agent, seed, "baseline", "custom"))
            
    num_workers = min(mp.cpu_count(), 16)
    with mp.Pool(num_workers) as pool:
        matches = pool.starmap(run_single_match, pool_args)
        
    valid_matches = [m for m in matches if m is not None]
    
    for i, m in enumerate(valid_matches):
        seed = m['seed']
        winner = m['winner']
        
        if seed % 2 == 0:
            # Custom = player 0
            if winner == 0:
                wins_custom += 1
            elif winner == 1:
                wins_baseline += 1
            else:
                draws += 1
            reserves.append(m['metrics_a']['reserve_ratio'])
            attacks.append(m['metrics_a']['attack_pct'])
            reinforces.append(m['metrics_a']['reinforce_pct'])
        else:
            # Custom = player 1
            if winner == 1:
                wins_custom += 1
            elif winner == 0:
                wins_baseline += 1
            else:
                draws += 1
            reserves.append(m['metrics_b']['reserve_ratio'])
            attacks.append(m['metrics_b']['attack_pct'])
            reinforces.append(m['metrics_b']['reinforce_pct'])
            
    total = len(valid_matches)
    win_rate = wins_custom / max(1, total)
    
    return {
        'win_rate': win_rate,
        'wins': wins_custom,
        'losses': wins_baseline,
        'draws': draws,
        'total': total,
        'avg_reserve': np.mean(reserves),
        'avg_attack': np.mean(attacks),
        'avg_reinforce': np.mean(reinforces),
    }

def main():
    print("Starting parameter search for ELO 1245 optimization...")
    
    # We will test a set of configurations to improve aggressiveness and efficiency
    configs = [
        # Baseline reference (approx)
        {
            'name': 'Baseline Reference',
            'params': {}
        },
        # Config 1: Lower ROI threshold (more aggressive attacks)
        {
            'name': 'ROI 1.1',
            'params': {'roi_threshold': 1.1}
        },
        # Config 2: Lower ROI + Higher Regroup Threshold (reduces passive shuffling)
        {
            'name': 'ROI 1.1 + Regroup 1.5',
            'params': {'roi_threshold': 1.1, 'regroup_pressure_delta_min': 1.5}
        },
        # Config 3: Regroup 2.0 (heavily restricts passive shuffling)
        {
            'name': 'Regroup 2.0 Only',
            'params': {'regroup_pressure_delta_min': 2.0}
        },
        # Config 4: ROI 1.2 + Regroup 1.5 (moderate aggressiveness)
        {
            'name': 'ROI 1.2 + Regroup 1.5',
            'params': {'roi_threshold': 1.2, 'regroup_pressure_delta_min': 1.5}
        },
        # Config 5: ROI 1.0 + Regroup 2.0 (highly aggressive)
        {
            'name': 'ROI 1.0 + Regroup 2.0',
            'params': {'roi_threshold': 1.0, 'regroup_pressure_delta_min': 2.0}
        }
    ]
    
    results = []
    for c in configs:
        print(f"\nEvaluating configuration: {c['name']}...")
        safe_name = "agent_" + c['name'].replace(' ', '_').replace('.', '_').replace('+', '_')
        if c['name'] == 'Baseline Reference':
            # Baseline vs Baseline matches are just 50% win rate by definition,
            # but we run it to collect baseline metrics on reserve/attacks/etc.
            res = run_evaluation(c['params'], safe_name, num_games=20)
            res['win_rate'] = 0.50 # fixed
        else:
            res = run_evaluation(c['params'], safe_name, num_games=30)
            
        print(f"  Win Rate: {res['win_rate']*100:.1f}% ({res['wins']} W - {res['losses']} L - {res['draws']} D)")
        print(f"  Avg Reserve: {res['avg_reserve']*100:.2f}%")
        print(f"  Avg Attack: {res['avg_attack']*100:.2f}%")
        print(f"  Avg Reinforce/Regroup: {res['avg_reinforce']*100:.2f}%")
        
        results.append({
            'name': c['name'],
            'win_rate': res['win_rate'],
            'wins': res['wins'],
            'losses': res['losses'],
            'draws': res['draws'],
            'avg_reserve': res['avg_reserve'],
            'avg_attack': res['avg_attack'],
            'avg_reinforce': res['avg_reinforce'],
            'params': c['params']
        })
        
    print("\n=== Optimization Summary ===")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save the optimization log to artifacts
    artifact_dir = "/home/yahor/.gemini/antigravity-cli/brain/fa8225ec-c931-461c-a04e-13f5bbf39638"
    df.to_json(os.path.join(artifact_dir, "optimization_results.json"), orient="records", indent=2)
    
    # Find the best configuration
    best = df.sort_values(by="win_rate", ascending=False).iloc[0]
    print(f"\nBest Config Found: {best['name']} with Win Rate {best['win_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
