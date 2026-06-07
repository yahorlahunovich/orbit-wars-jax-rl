import jax
import jax.numpy as jnp
from orbit_wars.state import OrbitWarsState
from orbit_wars.reset import reset
from orbit_wars.producer import project_garrison, _flow_terms_per_planet, competitive_score

def test_producer_logic():
    # 1. Setup a simple state
    state = reset(0)
    print(f"Step: {state.step}, Planets: {state.planets.shape[0]}")
    
    # 2. Project Garrison (Do-nothing)
    horizon = 20
    status = project_garrison(state, horizon)
    print(f"Status ships shape: {status.ships.shape}") # Should be [P, H+1]
    
    # 3. Calculate Flow Terms
    prod = state.planets[:, 6]
    produced, lost = _flow_terms_per_planet(status, prod)
    print(f"Produced per player: {produced}")
    print(f"Lost per player: {lost}")
    
    # 4. Competitive Score
    roi0 = competitive_score(produced, lost, 0)
    roi1 = competitive_score(produced, lost, 1)
    print(f"ROI Player 0: {roi0}")
    print(f"ROI Player 1: {roi1}")

if __name__ == "__main__":
    test_producer_logic()
