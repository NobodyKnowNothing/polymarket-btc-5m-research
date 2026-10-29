from typing import Callable
import numpy as np

def predict(get_features: Callable[[], list[dict]], sleep: Callable[[float], None]) -> None:
    """
    Delta-OBI Absorption:
    - Monitors the relationship between aggressive trade flow (Signed Delta) 
      and the liquidity structure (OBI).
    - A "divergence" (e.g., positive Delta but negative OBI) suggests that 
      aggressive buyers are being absorbed by large limit sell orders.
    - This absorption signal is used to bet against the current delta direction.
    """
    # Short sleep to confirm the divergence isn't a single-tick outlier
    sleep(15.0)
    
    feats = get_features()
    if not feats:
        return "Up"

    avg_delta = np.mean([f.get("Signed Delta", 0.0) for f in feats]) if feats else 0.0
    avg_obi = np.mean([f.get("OBI", 0.0) for f in feats]) if feats else 0.0

    # If delta and OBI have opposite signs, it indicates absorption.
    # Example: avg_delta is +50 (buying), but avg_obi is -0.1 (heavy sell side).
    # This often leads to a price drop.
    
    if abs(avg_delta) > 1.0:  # Ensure there is significant flow to analyze
        if np.sign(avg_delta) != np.sign(avg_obi):
            # Absorption detected: Bet against the aggressive flow
            return "Down" if avg_delta > 0 else "Up"
        else:
            # Flow and liquidity are aligned: Trend following
            return "Up" if avg_delta > 0 else "Down"
    
    # Default to OBI if delta is negligible
    return "Up" if avg_obi >= 0 else "Down"