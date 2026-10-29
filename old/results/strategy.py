from typing import Callable
import numpy as np

def predict(get_features: Callable[[], list[dict]], sleep: Callable[[float], None]) -> str:
    """
    Temporal Delta Velocity:
    - Samples features at two different time points.
    - Calculates the 'velocity' (change) of OBI and Signed Delta.
    - Bets on the direction of the acceleration, as it indicates increasing 
      aggressive pressure.
    """
    # First Snapshot
    feat1 = get_features()
    if not feat1:
        return "Up"
    
    # Poll for 15 seconds to measure velocity
    sleep(15.0)
    
    # Second Snapshot
    feat2 = get_features()
    if not feat2:
        return "Up"

    # Map exchanges to their features for comparison
    map1 = {f['exchange']: f for f in feat1}
    map2 = {f['exchange']: f for f in feat2}
    
    velocities = []
    for exch in map2:
        if exch in map1:
            f1 = map1[exch]
            f2 = map2[exch]
            
            # Calculate change in OBI and Delta
            delta_vel = f2.get("Signed Delta", 0.0) - f1.get("Signed Delta", 0.0)
            obi_vel = f2.get("OBI", 0.0) - f1.get("OBI", 0.0)
            
            # Combine velocity: Delta is more predictive of immediate moves
            combined_vel = (delta_vel * 0.7) + (obi_vel * 0.3)
            velocities.append(combined_vel)

    if not velocities:
        return "Up"

    avg_velocity = np.mean(velocities)
    # We look for a significant shift in momentum
    return "Up" if avg_velocity > 0 else "Down"