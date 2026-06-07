import os
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic")

from src.game import parse_state, Planet, Fleet, GameState
from src.geometry import fleet_speed, fleet_ray_closest_to_point
from src.strategy import aggregate_threats_by_planet, DEFAULT_CONFIG
from direct_runner import run_direct_from_names

def fleet_target_planet(fleet, planets, slop=2.0):
    best = None
    sp = fleet_speed(fleet.ships)
    ux = math.cos(fleet.angle)
    uy = math.sin(fleet.angle)
    
    for p in planets:
        t_close, d_close = fleet_ray_closest_to_point(fleet.x, fleet.y, sp, ux, uy, p.x, p.y)
        if t_close <= 0:
            continue
        if d_close <= p.radius + slop:
            if best is None or t_close < best[0]:
                best = (t_close, p.id)
    if best is None:
        return None
    return best[1]

def parse_game_steps(steps, p, label):
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
        
        # We need a dict observation format for parse_state
        if isinstance(obs, dict):
            obs_copy = dict(obs)
        else:
            obs_copy = {
                "planets": getattr(obs, "planets", []),
                "fleets": getattr(obs, "fleets", []),
                "player": getattr(obs, "player", 0),
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
        
    for t in range(1, T):
        state = states[t]
        my_planets_list = [planet for planet in state.planets if planet.owner == p]
        if not my_planets_list:
            continue
            
        my_ships_planets = sum(planet.ships for planet in my_planets_list)
        my_ships_fleets = sum(f.ships for f in state.fleets if f.owner == p)
        my_ships_total = my_ships_planets + my_ships_fleets
        
        my_planets = len(my_planets_list)
        enemy_planets = sum(1 for planet in state.planets if planet.owner >= 0 and planet.owner != p)
        neutral_planets = sum(1 for planet in state.planets if planet.owner == -1)
        
        action = actions[t]
        num_launches = 0
        ships_launched = 0
        ships_expansion = 0
        ships_attack = 0
        ships_reinforce = 0
        
        if action and isinstance(action, list):
            planet_owners = {pl.id: pl.owner for pl in state.planets}
            planet_lookup = {pl.id: pl for pl in state.planets}
            for move in action:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                ships = int(ships)
                if from_id not in planet_lookup:
                    continue
                from_planet = planet_lookup[from_id]
                num_launches += 1
                ships_launched += ships
                
                # Target estimation
                ux = math.cos(angle)
                uy = math.sin(angle)
                start_x = from_planet.x + ux * (from_planet.radius + 0.1)
                start_y = from_planet.y + uy * (from_planet.radius + 0.1)
                dummy_fleet = Fleet(id=-1, owner=p, x=start_x, y=start_y, angle=angle, from_planet_id=from_id, ships=ships)
                
                tgt_id = fleet_target_planet(dummy_fleet, state.planets)
                if tgt_id is not None:
                    tgt_owner = planet_owners.get(tgt_id, -1)
                    if tgt_owner == -1:
                        ships_expansion += ships
                    elif tgt_owner == p:
                        ships_reinforce += ships
                    else:
                        ships_attack += ships
                        
        records.append({
            'label': label,
            'step': t,
            'my_ships_total': my_ships_total,
            'my_ships_planets': my_ships_planets,
            'my_planets': my_planets,
            'enemy_planets': enemy_planets,
            'neutral_planets': neutral_planets,
            'num_launches': num_launches,
            'ships_launched': ships_launched,
            'ships_expansion': ships_expansion,
            'ships_attack': ships_attack,
            'ships_reinforce': ships_reinforce,
            'reserve_ratio': my_ships_planets / max(1.0, my_ships_total)
        })
    return records

def process_top_game(args):
    filepath, folder, episode_id = args
    if not os.path.exists(filepath):
        parent = os.path.dirname(filepath)
        filepath = os.path.join(parent, f"episode-{episode_id}.json")
        if not os.path.exists(filepath):
            return []
            
    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            return []
            
    steps = data.get("steps", [])
    if len(steps) < 100:
        return []
        
    # In ELO >= 1300 games, both players are top ELO, so we analyze both
    all_records = []
    for p in [0, 1]:
        all_records.extend(parse_game_steps(steps, p, "Top ELO Replays (>=1300)"))
    return all_records

def main():
    base_dir = "/media/yahor/ADATA SE880/datasets/orbit-wars/replays_raw"
    folders = ["all1", "all2", "all3"]
    
    # 1. Load Top ELO Manifest entries
    dfs = []
    for folder in folders:
        csv_path = os.path.join(base_dir, folder, "manifest.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["folder"] = folder
            dfs.append(df)
            
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["agent_count"] == 2]
    # Absolute top tier: avg_score >= 1300 ELO
    top_manifest = combined[combined["avg_score"] >= 1300]
    
    print(f"Total ELO >=1300 games available: {len(top_manifest)}")
    
    # Sample 30 games
    np.random.seed(42)
    sample_size = 30
    sampled_top = top_manifest.sample(n=min(sample_size, len(top_manifest)))
    
    tasks = []
    for _, row in sampled_top.iterrows():
        filepath = os.path.join(base_dir, row['folder'], f"{row['episode_id']}.json")
        tasks.append((filepath, row['folder'], row['episode_id']))
        
    print(f"Parsing {len(tasks)} top ELO replays in parallel...")
    num_workers = min(mp.cpu_count(), 16)
    with mp.Pool(num_workers) as pool:
        top_results_nested = pool.map(process_top_game, tasks)
        
    top_records = []
    for res in top_results_nested:
        top_records.extend(res)
        
    # 2. Run 30 games of Simplified_1245 vs kaggle700_current_heuristic locally
    print("\nRunning 30 games of Simplified_1245 vs kaggle700 locally...")
    agent_a_path = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/heuristics/Simplified_Orbit_Wars_Agent_1245/main.py"
    agent_b_path = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic/main.py"
    
    local_records = []
    for seed in range(100, 130):
        # We need to run with keep_steps=True to get the full trajectory
        steps, elapsed = run_direct_from_names(
            [agent_a_path, agent_b_path],
            root=Path("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2"),
            seed=seed,
            episode_steps=500,
            keep_steps=True
        )
        # Parse from perspective of agent A (player 0)
        local_records.extend(parse_game_steps(steps, 0, "Simplified_1245 Bot"))
        
    print("Parsing completed.")
    
    # 3. Combine DataFrames and Analyze Gaps
    df_top = pd.DataFrame(top_records)
    df_local = pd.DataFrame(local_records)
    df = pd.concat([df_top, df_local], ignore_index=True)
    
    # Output statistics comparison
    print("\n=== Trajectory Parameter Comparison ===")
    
    # 1) Opening Planet Expansion (turns 0-80)
    top_exp = df_top[df_top['step'] == 80]['my_planets'].mean()
    local_exp = df_local[df_local['step'] == 80]['my_planets'].mean()
    print(f"Average Planets Owned at Turn 80:")
    print(f"  Top ELO Replays: {top_exp:.2f}")
    print(f"  Simplified_1245: {local_exp:.2f}")
    print(f"  Gap: {local_exp - top_exp:+.2f} planets")
    
    # 2) Garrison Reserve Ratio (turns 100-400)
    top_res = df_top[df_top['step'].between(100, 400)]['reserve_ratio'].mean()
    local_res = df_local[df_local['step'].between(100, 400)]['reserve_ratio'].mean()
    print(f"\nAverage Planet Garrison Reserve Ratio (Turns 100-400):")
    print(f"  Top ELO Replays: {top_res * 100:.2f}%")
    print(f"  Simplified_1245: {local_res * 100:.2f}%")
    print(f"  Gap: {(local_res - top_res)*100:+.2f}%")
    
    # 3) Launch mix percentages
    top_launch_sum = df_top['ships_launched'].sum()
    print(f"\nTop ELO Replays Launch Mix:")
    print(f"  Expansion: {df_top['ships_expansion'].sum() / top_launch_sum * 100:.2f}%")
    print(f"  Attack: {df_top['ships_attack'].sum() / top_launch_sum * 100:.2f}%")
    print(f"  Reinforce/Defense: {df_top['ships_reinforce'].sum() / top_launch_sum * 100:.2f}%")
    
    local_launch_sum = df_local['ships_launched'].sum()
    print(f"Simplified_1245 Bot Launch Mix:")
    print(f"  Expansion: {df_local['ships_expansion'].sum() / local_launch_sum * 100:.2f}%")
    print(f"  Attack: {df_local['ships_attack'].sum() / local_launch_sum * 100:.2f}%")
    print(f"  Reinforce/Defense: {df_local['ships_reinforce'].sum() / local_launch_sum * 100:.2f}%")
    
    # 4. Save plots
    artifact_dir = "/home/yahor/.gemini/antigravity-cli/brain/fa8225ec-c931-461c-a04e-13f5bbf39638"
    sns.set_theme(style="whitegrid")
    
    # Planet ownership trajectories comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='my_planets', hue='label', errorbar=('ci', 95), linewidth=2.5)
    plt.title('Planet Expansion Comparison: Simplified_1245 vs. Top ELO', fontsize=14, pad=15)
    plt.xlabel('Game Step (Turn)', fontsize=12)
    plt.ylabel('Planets Owned', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "comparison_planets.png"), dpi=150)
    plt.close()
    
    # Garrison Reserve comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='reserve_ratio', hue='label', errorbar=('ci', 95), linewidth=2.5)
    plt.title('Planet Garrison Reserve Ratio Comparison', fontsize=14, pad=15)
    plt.xlabel('Game Step (Turn)', fontsize=12)
    plt.ylabel('Reserve Ratio (Ships on Planets / Total Ships)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "comparison_reserves.png"), dpi=150)
    plt.close()
    
    # Save statistics details
    comparison_stats = {
        'top_planets_t80': float(top_exp),
        'local_planets_t80': float(local_exp),
        'top_reserve_ratio': float(top_res),
        'local_reserve_ratio': float(local_res),
        'top_expansion_pct': float(df_top['ships_expansion'].sum() / top_launch_sum),
        'top_attack_pct': float(df_top['ships_attack'].sum() / top_launch_sum),
        'top_reinforce_pct': float(df_top['ships_reinforce'].sum() / top_launch_sum),
        'local_expansion_pct': float(df_local['ships_expansion'].sum() / local_launch_sum),
        'local_attack_pct': float(df_local['ships_attack'].sum() / local_launch_sum),
        'local_reinforce_pct': float(df_local['ships_reinforce'].sum() / local_launch_sum)
    }
    
    with open(os.path.join(artifact_dir, "gap_analysis_summary.json"), "w") as f:
        json.dump(comparison_stats, f, indent=2)
        
    print("\nGap analysis visualizations saved successfully.")

if __name__ == "__main__":
    main()
