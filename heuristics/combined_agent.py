import os
import sys
import math
from pathlib import Path

# Add this folder and the geometry library folder to sys.path
agent_dir = str(Path(__file__).resolve().parent)
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

geom_dir = "/home/yahor/Documents/coding/orbit-wars/orbit_wars_cursor_template_v2/versions/kaggle700_current_heuristic"
if geom_dir not in sys.path:
    sys.path.insert(0, geom_dir)

from src.geometry import fleet_speed, fleet_ray_closest_to_point

# Import both agents
from Agent_1110.main import agent as agent_1110
from Simplified_Orbit_Wars_Agent_1245.main import agent as agent_1245

def get_target_planet_id(from_id, angle, ships, planets_list):
    from_p = None
    for p in planets_list:
        if int(p[0]) == from_id:
            from_p = p
            break
    if from_p is None:
        return None
        
    fx = float(from_p[2]) + math.cos(angle) * (float(from_p[4]) + 0.1)
    fy = float(from_p[3]) + math.sin(angle) * (float(from_p[4]) + 0.1)
    
    sp = fleet_speed(ships)
    ux = math.cos(angle)
    uy = math.sin(angle)
    
    best = None
    for p in planets_list:
        t_close, d_close = fleet_ray_closest_to_point(fx, fy, sp, ux, uy, float(p[2]), float(p[3]))
        if t_close <= 0:
            continue
        if d_close <= float(p[4]) + 2.0:
            if best is None or t_close < best[0]:
                best = (t_close, int(p[0]))
    if best is None:
        return None
    return best[1]

def agent(obs):
    try:
        # Standardize obs format for both agents (ensure dict for 1245)
        if not isinstance(obs, dict):
            obs_dict = {
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
        else:
            obs_dict = obs

        # 1. Run both agents
        moves_1110 = agent_1110(obs)
        moves_1245 = agent_1245(obs_dict)
        
        planets_data = obs_dict["planets"]
        player_id = int(obs_dict["player"])
        
        # Available ships on our planets
        avail_ships = {}
        planet_owners = {}
        for p in planets_data:
            pid = int(p[0])
            owner = int(p[1])
            ships = int(p[5])
            planet_owners[pid] = owner
            if owner == player_id:
                avail_ships[pid] = ships
                
        combined_moves = []
        
        # 2. First pass: Prioritize defensive/reinforcement moves from Agent_1110
        # Agent_1110 is highly simulator-driven and excellent at threat defense
        for move in moves_1110:
            if len(move) != 3:
                continue
            from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            
            tgt_id = get_target_planet_id(from_id, angle, ships, planets_data)
            if tgt_id is not None and planet_owners.get(tgt_id) == player_id:
                # This is a defensive reinforcement move
                if avail_ships.get(from_id, 0) >= ships:
                    combined_moves.append([from_id, angle, ships])
                    avail_ships[from_id] -= ships
                    
        # 3. Second pass: Run all expansion and attack moves from Simplified_Agent_1245 (strongest overall strategy)
        for move in moves_1245:
            if len(move) != 3:
                continue
            from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            if avail_ships.get(from_id, 0) >= ships:
                combined_moves.append([from_id, angle, ships])
                avail_ships[from_id] -= ships
                
        # 4. Third pass: Any remaining offensive/expansion moves from Agent_1110 if we still have surplus ships
        for move in moves_1110:
            if len(move) != 3:
                continue
            from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
            
            # Check if this move was already added in the first pass
            already_added = any(m[0] == from_id and abs(m[1] - angle) < 1e-4 and m[2] == ships for m in combined_moves)
            if already_added:
                continue
                
            if avail_ships.get(from_id, 0) >= ships:
                combined_moves.append([from_id, angle, ships])
                avail_ships[from_id] -= ships
                
        return combined_moves
    except Exception:
        # Fallback to the best single agent
        try:
            return agent_1245(obs)
        except Exception:
            return []
