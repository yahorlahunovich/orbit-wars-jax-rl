import os
import json
import math
import sys
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

sys.path.append("/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic")

from src.game import parse_state, Planet, Fleet, GameState
from src.geometry import fleet_speed, fleet_ray_closest_to_point
from src.strategy import aggregate_threats_by_planet, DEFAULT_CONFIG

FEATURE_KEYS = [
    'step_norm',
    'my_ships_log',
    'enemy_ships_log',
    'ship_ratio',
    'my_planets',
    'enemy_planets',
    'neutral_planets',
    'my_prod',
    'enemy_prod',
    'prod_ratio',
    'planet_ratio',
    'current_threat_log',
    'threat_ratio',
    'avg_dist_to_enemy',
    'avg_dist_to_neutral'
]

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
    return best[0], best[1] # eta, target_id

def process_single_replay(filepath):
    results = []
    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            return []
            
    steps = data.get("steps", [])
    T = len(steps)
    if T < 100:
        return []
        
    for p in [0, 1]:
        # Pre-parse all states
        states = []
        for t in range(T):
            obs = steps[t][p]['observation']
            obs_copy = dict(obs)
            obs_copy['player'] = p
            state = parse_state(obs_copy)
            states.append(state)
            
        for t in range(1, T - 50):
            state = states[t]
            my_planets_list = [planet for planet in state.planets if planet.owner == p]
            if not my_planets_list:
                continue
                
            enemy_planets_list = [planet for planet in state.planets if planet.owner == 1 - p]
            neutral_planets_list = [planet for planet in state.planets if planet.owner == -1]
            
            # Calculate features
            my_ships = sum(planet.ships for planet in state.planets if planet.owner == p) + \
                       sum(f.ships for f in state.fleets if f.owner == p)
            enemy_ships = sum(planet.ships for planet in state.planets if planet.owner == 1 - p) + \
                          sum(f.ships for f in state.fleets if f.owner == 1 - p)
            
            my_planets = len(my_planets_list)
            enemy_planets = len(enemy_planets_list)
            neutral_planets = len(neutral_planets_list)
            
            my_prod = sum(planet.production for planet in my_planets_list)
            enemy_prod = sum(planet.production for planet in enemy_planets_list)
            
            threats_by_planet = aggregate_threats_by_planet(state, DEFAULT_CONFIG)
            current_threat = sum(f.ships for threats in threats_by_planet.values() for eta, f in threats)
            
            if my_planets_list and enemy_planets_list:
                avg_dist_to_enemy = sum(math.hypot(p_node.x - e_node.x, p_node.y - e_node.y) for p_node in my_planets_list for e_node in enemy_planets_list) / (my_planets * enemy_planets)
            else:
                avg_dist_to_enemy = 100.0
                
            if my_planets_list and neutral_planets_list:
                avg_dist_to_neutral = sum(math.hypot(p_node.x - n_node.x, p_node.y - n_node.y) for p_node in my_planets_list for n_node in neutral_planets_list) / (my_planets * neutral_planets)
            else:
                avg_dist_to_neutral = 100.0
                
            features = {
                'step_norm': t / 500.0,
                'my_ships_log': math.log(my_ships + 1),
                'enemy_ships_log': math.log(enemy_ships + 1),
                'ship_ratio': my_ships / max(1.0, enemy_ships),
                'my_planets': my_planets,
                'enemy_planets': enemy_planets,
                'neutral_planets': neutral_planets,
                'my_prod': my_prod,
                'enemy_prod': enemy_prod,
                'prod_ratio': my_prod / max(1.0, enemy_prod),
                'planet_ratio': my_planets / max(1.0, enemy_planets),
                'current_threat_log': math.log(current_threat + 1),
                'threat_ratio': current_threat / max(1.0, my_ships),
                'avg_dist_to_enemy': avg_dist_to_enemy,
                'avg_dist_to_neutral': avg_dist_to_neutral
            }
            
            # Future quantities [t+1, t+50]
            total_enemy_ships_sent = 0
            total_neutral_ships_sent = 0
            total_reinforce_ships_sent = 0
            max_threat_encountered = 0
            
            for t_future in range(t + 1, t + 51):
                future_state = states[t_future]
                f_threats_by_planet = aggregate_threats_by_planet(future_state, DEFAULT_CONFIG)
                f_threat = sum(f.ships for threats in f_threats_by_planet.values() for eta, f in threats)
                max_threat_encountered = max(max_threat_encountered, f_threat)
                
                action = steps[t_future][p].get('action', [])
                if action and isinstance(action, list):
                    planet_owners = {pl.id: pl.owner for pl in future_state.planets}
                    planet_lookup = {pl.id: pl for pl in future_state.planets}
                    
                    for move in action:
                        if len(move) != 3:
                            continue
                        from_id, angle, ships = move
                        ships = int(ships)
                        if from_id not in planet_lookup:
                            continue
                        from_planet = planet_lookup[from_id]
                        
                        ux = math.cos(angle)
                        uy = math.sin(angle)
                        start_x = from_planet.x + ux * (from_planet.radius + 0.1)
                        start_y = from_planet.y + uy * (from_planet.radius + 0.1)
                        dummy_fleet = Fleet(id=-1, owner=p, x=start_x, y=start_y, angle=angle, from_planet_id=from_id, ships=ships)
                        
                        res = fleet_target_planet(dummy_fleet, future_state.planets)
                        if res is not None:
                            eta, tgt_id = res
                            tgt_owner = planet_owners.get(tgt_id, -1)
                            if tgt_owner == -1:
                                total_neutral_ships_sent += ships
                            elif tgt_owner == 1 - p:
                                total_enemy_ships_sent += ships
                            elif tgt_owner == p:
                                total_reinforce_ships_sent += ships
                                
            # Define label
            total_sent = total_enemy_ships_sent + total_neutral_ships_sent + total_reinforce_ships_sent
            if total_sent < 30:
                label = 'GROW'
            else:
                if total_reinforce_ships_sent > 0.4 * total_sent and total_reinforce_ships_sent > 15:
                    label = 'DEFENSE'
                elif total_enemy_ships_sent > total_neutral_ships_sent:
                    label = 'ATTACK'
                else:
                    label = 'EXPANSION'
                    
            results.append((features, label))
            
    return results

def main():
    replay_dir = "/media/yahor/ADATA SE880/datasets/orbit-wars/replays_raw/top_players"
    files = [f for f in os.listdir(replay_dir) if f.endswith(".json")]
    # Process 250 files for robust dataset
    num_files = min(250, len(files))
    selected_files = sorted(files)[:num_files]
    filepaths = [os.path.join(replay_dir, f) for f in selected_files]
    
    print(f"Extracting features from {num_files} replays in parallel...")
    
    num_workers = min(mp.cpu_count(), 16)
    with mp.Pool(num_workers) as pool:
        results_nested = pool.map(process_single_replay, filepaths)
        
    all_samples = []
    for res in results_nested:
        all_samples.extend(res)
        
    print(f"Extracted {len(all_samples)} training samples.")
    
    # Separate features and labels
    X_dict = [s[0] for s in all_samples]
    y_str = [s[1] for s in all_samples]
    
    # Convert features to numpy array
    X = np.zeros((len(X_dict), len(FEATURE_KEYS)), dtype=np.float32)
    for i, feat in enumerate(X_dict):
        for j, key in enumerate(FEATURE_KEYS):
            X[i, j] = feat[key]
            
    # Class mapping
    class_names = ['EXPANSION', 'DEFENSE', 'ATTACK', 'GROW']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([class_to_idx[name] for name in y_str], dtype=np.int32)
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardization (Z-score)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0 # Avoid division by zero
    
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    
    print("Training Multi-Layer Perceptron (MLP) Stage Classifier...")
    # Small hidden layer size for lightweight deployment (32 nodes, 16 nodes)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = mlp.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Export model parameters to a python file
    # We will write these parameters to models/stage_classifier.json
    export_dir = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/models"
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "stage_classifier.json")
    
    model_data = {
        'feature_keys': FEATURE_KEYS,
        'class_names': class_names,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'coefs': [w.tolist() for w in mlp.coefs_],
        'intercepts': [b.tolist() for b in mlp.intercepts_]
    }
    
    with open(export_path, "w") as f:
        json.dump(model_data, f, indent=2)
        
    print(f"Model exported successfully to {export_path}")
    
    # Also generate a python module file containing model parameters for easy direct import
    py_module_path = os.path.join(export_dir, "stage_classifier_model.py")
    with open(py_module_path, "w") as f:
        f.write("# Auto-generated model file. Do not edit.\n")
        f.write("import numpy as np\n\n")
        f.write(f"FEATURE_KEYS = {repr(FEATURE_KEYS)}\n")
        f.write(f"CLASS_NAMES = {repr(class_names)}\n\n")
        f.write(f"MEAN = np.array({repr(mean.tolist())}, dtype=np.float32)\n")
        f.write(f"STD = np.array({repr(std.tolist())}, dtype=np.float32)\n\n")
        f.write("COEFS = [\n")
        for w in mlp.coefs_:
            f.write(f"    np.array({repr(w.tolist())}, dtype=np.float32),\n")
        f.write("]\n\n")
        f.write("INTERCEPTS = [\n")
        for b in mlp.intercepts_:
            f.write(f"    np.array({repr(b.tolist())}, dtype=np.float32),\n")
        f.write("]\n\n")
        f.write("""def predict_stage(features_dict):
    # Prepare features
    feats = np.zeros(len(FEATURE_KEYS), dtype=np.float32)
    for idx, key in enumerate(FEATURE_KEYS):
        feats[idx] = features_dict.get(key, 0.0)
        
    # Scale features
    x = (feats - MEAN) / STD
    
    # Forward pass
    for W, b in zip(COEFS[:-1], INTERCEPTS[:-1]):
        x = np.maximum(0.0, np.dot(x, W) + b) # ReLU activation
        
    logits = np.dot(x, COEFS[-1]) + INTERCEPTS[-1]
    pred_idx = int(np.argmax(logits))
    return CLASS_NAMES[pred_idx]
""")
    print(f"Python module helper exported to {py_module_path}")

if __name__ == "__main__":
    main()
