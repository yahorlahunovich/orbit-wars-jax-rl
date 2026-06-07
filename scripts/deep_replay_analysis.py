import os
import json
import math
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic")

from src.game import parse_state, Planet, Fleet, GameState
from src.geometry import fleet_speed, fleet_ray_closest_to_point
from src.strategy import aggregate_threats_by_planet, DEFAULT_CONFIG

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
    return best[1] # target planet id

def process_single_game(args):
    filepath, folder, episode_id, elo_tier = args
    if not os.path.exists(filepath):
        # Try alternate path
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
    T = len(steps)
    if T < 100:
        return []
        
    game_records = []
    
    # We analyze from the perspective of both players in 2-player games
    # In 4-player games, we analyze all active players
    agent_count = len(steps[0])
    
    for p in range(agent_count):
        # Determine player's final outcome (did they win?)
        # For Kaggle games, final reward or score determines who won.
        # We can extract the final rewards from data['rewards'] or steps[-1][p]['reward']
        rewards = data.get("rewards", [0]*agent_count)
        if rewards is None or len(rewards) < agent_count:
            rewards = [step_info.get('reward', 0) for step_info in steps[-1]]
            
        is_winner = False
        if rewards and max(rewards) > min(rewards):
            is_winner = (rewards[p] == max(rewards))
            
        # Pre-parse all states
        states = []
        for t in range(T):
            obs = steps[t][p]['observation']
            obs_copy = dict(obs)
            obs_copy['player'] = p
            state = parse_state(obs_copy)
            states.append(state)
            
        for t in range(1, T):
            state = states[t]
            my_planets_list = [planet for planet in state.planets if planet.owner == p]
            if not my_planets_list:
                # Player is dead or has no planets
                continue
                
            my_ships_planets = sum(planet.ships for planet in my_planets_list)
            my_ships_fleets = sum(f.ships for f in state.fleets if f.owner == p)
            my_ships_total = my_ships_planets + my_ships_fleets
            
            enemy_planets_list = [planet for planet in state.planets if planet.owner >= 0 and planet.owner != p]
            enemy_ships_total = sum(planet.ships for planet in enemy_planets_list) + \
                                sum(f.ships for f in state.fleets if f.owner >= 0 and f.owner != p)
                                
            my_planets = len(my_planets_list)
            enemy_planets = len(enemy_planets_list)
            neutral_planets = sum(1 for planet in state.planets if planet.owner == -1)
            
            # Action launch details
            action = steps[t][p].get('action', [])
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
                            
            # Threats
            threats_by_planet = aggregate_threats_by_planet(state, DEFAULT_CONFIG)
            current_threat = sum(f.ships for threats in threats_by_planet.values() for eta, f in threats)
            
            record = {
                'episode_id': episode_id,
                'elo_tier': elo_tier,
                'player_id': p,
                'is_winner': is_winner,
                'step': t,
                'my_ships_total': my_ships_total,
                'my_ships_planets': my_ships_planets,
                'my_ships_fleets': my_ships_fleets,
                'enemy_ships_total': enemy_ships_total,
                'my_planets': my_planets,
                'enemy_planets': enemy_planets,
                'neutral_planets': neutral_planets,
                'current_threat': current_threat,
                'num_launches': num_launches,
                'ships_launched': ships_launched,
                'ships_expansion': ships_expansion,
                'ships_attack': ships_attack,
                'ships_reinforce': ships_reinforce
            }
            game_records.append(record)
            
    return game_records

def main():
    base_dir = "/media/yahor/ADATA SE880/datasets/orbit-wars/replays_raw"
    folders = ["all1", "all2", "all3"]
    
    # 1. Read manifests and group by ELO
    dfs = []
    for folder in folders:
        csv_path = os.path.join(base_dir, folder, "manifest.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["folder"] = folder
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    # Filter to 2-player games for consistent analysis
    combined = combined[combined["agent_count"] == 2]
    
    # Classify into Low, Mid, High ELO
    # Low: < 700 ELO, Mid: 700-1100 ELO, High: >= 1100 ELO
    low_elo = combined[combined["avg_score"] < 700]
    mid_elo = combined[combined["avg_score"].between(700, 1100)]
    high_elo = combined[combined["avg_score"] >= 1100]
    
    print(f"Dataset ELO Tiers (2-player games):")
    print(f"  Low ELO (<700): {len(low_elo)} games")
    print(f"  Mid ELO (700-1100): {len(mid_elo)} games")
    print(f"  High ELO (>=1100): {len(high_elo)} games")
    
    # Sample 40 games from each tier to get a solid distribution
    np.random.seed(42)
    sample_size = 40
    
    sampled_low = low_elo.sample(n=min(sample_size, len(low_elo)))
    sampled_mid = mid_elo.sample(n=min(sample_size, len(mid_elo)))
    sampled_high = high_elo.sample(n=min(sample_size, len(high_elo)))
    
    tasks = []
    for df_sampled, tier in [(sampled_low, 'Low ELO (<700)'), 
                             (sampled_mid, 'Mid ELO (700-1100)'), 
                             (sampled_high, 'High ELO (>=1100)')]:
        for _, row in df_sampled.iterrows():
            filepath = os.path.join(base_dir, row['folder'], f"{row['episode_id']}.json")
            tasks.append((filepath, row['folder'], row['episode_id'], tier))
            
    print(f"\nProcessing {len(tasks)} games in parallel...")
    
    num_workers = min(mp.cpu_count(), 16)
    with mp.Pool(num_workers) as pool:
        results_nested = pool.map(process_single_game, tasks)
        
    flat_records = []
    for res in results_nested:
        flat_records.extend(res)
        
    df = pd.DataFrame(flat_records)
    print(f"Analysis dataset contains {len(df)} row entries.")
    
    # 2. Plotting Visualizations
    artifact_dir = "/home/yahor/.gemini/antigravity-cli/brain/fa8225ec-c931-461c-a04e-13f5bbf39638"
    os.makedirs(artifact_dir, exist_ok=True)
    
    print("\nGenerating visualization plots...")
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Ship Count Trajectories over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='my_ships_total', hue='elo_tier', errorbar=('ci', 95), linewidth=2.5)
    plt.title('Average Total Ship Count Trajectory by ELO Tier', fontsize=14, pad=15)
    plt.xlabel('Game Step (Turn)', fontsize=12)
    plt.ylabel('Total Ships (Planets + Fleets)', fontsize=12)
    plt.legend(title='ELO Tier')
    plt.tight_layout()
    plot1_path = os.path.join(artifact_dir, "ship_count_trajectories.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved Plot 1: {plot1_path}")
    
    # PLOT 2: Planet Count Trajectories over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='my_planets', hue='elo_tier', errorbar=('ci', 95), linewidth=2.5)
    plt.title('Average Planet Ownership Trajectory by ELO Tier', fontsize=14, pad=15)
    plt.xlabel('Game Step (Turn)', fontsize=12)
    plt.ylabel('Planets Owned', fontsize=12)
    plt.legend(title='ELO Tier')
    plt.tight_layout()
    plot2_path = os.path.join(artifact_dir, "planet_ownership_trajectories.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved Plot 2: {plot2_path}")
    
    # PLOT 3: Action Launch Mix (Expansion, Attack, Defense) by ELO Tier
    # Group steps into early (0-100), mid (100-300), late (300-500)
    df['phase'] = pd.cut(df['step'], bins=[0, 100, 300, 500], labels=['Early (0-100)', 'Mid (100-300)', 'Late (300+)'])
    
    # Sum up launch categories
    mix_data = df.groupby(['elo_tier', 'phase'], observed=True)[['ships_expansion', 'ships_attack', 'ships_reinforce']].mean().reset_index()
    # Normalize mix_data
    mix_data['total_sent'] = mix_data['ships_expansion'] + mix_data['ships_attack'] + mix_data['ships_reinforce']
    mix_data['total_sent'] = mix_data['total_sent'].replace(0, 1.0)
    mix_data['Expansion %'] = mix_data['ships_expansion'] / mix_data['total_sent'] * 100
    mix_data['Attack %'] = mix_data['ships_attack'] / mix_data['total_sent'] * 100
    mix_data['Defense/Reinforce %'] = mix_data['ships_reinforce'] / mix_data['total_sent'] * 100
    
    # Reshape for bar plotting
    melted = pd.melt(mix_data, id_vars=['elo_tier', 'phase'], value_vars=['Expansion %', 'Attack %', 'Defense/Reinforce %'], 
                     var_name='Launch Type', value_name='Percentage')
    
    g = sns.catplot(
        data=melted, kind="bar",
        x="phase", y="Percentage", hue="Launch Type", col="elo_tier",
        palette="muted", height=5, aspect=1.0
    )
    g.set_axis_labels("Game Phase", "Percentage of Ships Launched (%)")
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle("Launch Strategy Distribution by ELO Tier & Game Phase", fontsize=15)
    plot3_path = os.path.join(artifact_dir, "launch_mix_distribution.png")
    g.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"Saved Plot 3: {plot3_path}")
    
    # PLOT 4: Defensive Reserve Ratio over time
    # Reserve ratio = ships on planets / total ships
    df['reserve_ratio'] = df['my_ships_planets'] / df['my_ships_total'].replace(0, 1.0)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='reserve_ratio', hue='elo_tier', errorbar=('ci', 95), linewidth=2.5)
    plt.title('Planet Garrison Reserve Ratio by ELO Tier', fontsize=14, pad=15)
    plt.xlabel('Game Step (Turn)', fontsize=12)
    plt.ylabel('Reserve Ratio (Ships on Planets / Total Ships)', fontsize=12)
    plt.legend(title='ELO Tier')
    plt.tight_layout()
    plot4_path = os.path.join(artifact_dir, "reserve_ratios.png")
    plt.savefig(plot4_path, dpi=150)
    plt.close()
    print(f"Saved Plot 4: {plot4_path}")
    
    # PLOT 5: Launch Frequency vs Launch Size
    # Calculate average launches per step and average ships per launch when launching
    launching_steps = df[df['num_launches'] > 0]
    launch_stats = launching_steps.groupby('elo_tier').agg(
        avg_ships_per_launch=('ships_launched', lambda x: (x / launching_steps.loc[x.index, 'num_launches']).mean()),
        avg_launches_per_turn=('num_launches', 'mean')
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    sns.barplot(data=launch_stats, x='elo_tier', y='avg_ships_per_launch', ax=ax1, color=color, alpha=0.7)
    ax1.set_ylabel('Average Ships per Launch (Size)', color=color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('ELO Tier', fontsize=12)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    sns.lineplot(data=launch_stats, x='elo_tier', y='avg_launches_per_turn', ax=ax2, color=color, marker='o', linewidth=3, markersize=10)
    ax2.set_ylabel('Average Launches per Active Turn (Frequency)', color=color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Launch Size vs. Launch Frequency by ELO Tier', fontsize=14, pad=15)
    plt.tight_layout()
    plot5_path = os.path.join(artifact_dir, "launch_size_frequency.png")
    plt.savefig(plot5_path, dpi=150)
    plt.close()
    print(f"Saved Plot 5: {plot5_path}")
    
    # 3. Save Summary Stats to JSON
    summary_stats = {
        'low_elo_mean_ships': float(df[df['elo_tier'] == 'Low ELO (<700)']['my_ships_total'].mean()),
        'mid_elo_mean_ships': float(df[df['elo_tier'] == 'Mid ELO (700-1100)']['my_ships_total'].mean()),
        'high_elo_mean_ships': float(df[df['elo_tier'] == 'High ELO (>=1100)']['my_ships_total'].mean()),
        'low_elo_mean_planets': float(df[df['elo_tier'] == 'Low ELO (<700)']['my_planets'].mean()),
        'high_elo_mean_planets': float(df[df['elo_tier'] == 'High ELO (>=1100)']['my_planets'].mean()),
        'low_elo_reserve_ratio': float(df[df['elo_tier'] == 'Low ELO (<700)']['reserve_ratio'].mean()),
        'high_elo_reserve_ratio': float(df[df['elo_tier'] == 'High ELO (>=1100)']['reserve_ratio'].mean()),
    }
    
    with open(os.path.join(artifact_dir, "deep_analysis_summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)
    print("Saved summary statistics JSON.")

if __name__ == "__main__":
    main()
