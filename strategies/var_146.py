# Var 146 | BT: 0.627 | Live PnL: $-2.28 | Trades: 6
# Desc: Implements "Directional Coherence & Liquidity Quality (DCLQ)," which filters trades by requiring that the model's predicted direction is strictly corroborated by both aggressive order flow (Delta) and
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 2.651597, -0.9275333285331726, -1.0], "live_pnl": -2.2759363651275635, "live_wins": 1, "live_total": 6, "age": 225, "rounds_underwater": 6, "abstain_rounds": 59, "backtest_score": 0.6270300995920373, "backtest_acc": 0.5675413022351797, "backtest_n": 165}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between price movement and order flow intensity.
    """
    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Price Displacement: Signed distance from the window start price
    X['price_displacement'] = X['binance_price'] - X['strike_price']

    # 2. Flow-Price Coherence: Measures if delta and price movement are aligned
    # Positive if both are moving in the same direction, negative if opposing
    X['flow_price_coherence'] = X['mean_delta'] * np.sign(X['price_displacement'])

    # 3. Liquidity Quality: Normalized measure of market tightness
    # Higher value = lower spreads across OBI and price, indicating higher conviction/lower slippage
    X['liquidity_quality'] = 1.0 / (1.0 + X['obi_spread'] + X['price_spread'])

    # 4. Delta Intensity: Aggression relative to current volatility
    # Dampened by log to prevent extreme outliers from dominating
    X['delta_intensity'] = np.log1p(X['mean_delta'].abs() / (X['mean_volatility'] + 0.001))

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return robust, regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Directional Coherence & Liquidity Quality (DCLQ)".

    Logic:
    - Directional Coherence: We only enter a trade if the model's predicted side 
      matches the sign of BOTH the aggressive delta AND the price displacement.
      This avoids "Absorption" (high delta, no price move) and "Reversals" 
      (high delta, price moving opposite).
    - Dynamic Thresholding: Increases the required EV as 'seconds_remaining' 
      decreases to avoid the high-noise environment of the final 60 seconds.
    - Liquidity-Scaled Sizing: Scales the bet fraction based on the 'liquidity_quality' 
      and the 'delta_intensity' to ensure we only bet big when the move is "clean."
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        side_dir = 1
    else:
        best_ev = ev_down
        side = 'Down'
        side_dir = -1

    # Extract engineered features
    mean_delta = features_row.get('mean_delta', 0.0)
    price_disp = features_row.get('price_displacement', 0.0)
    lq = features_row.get('liquidity_quality', 0.5)
    di = features_row.get('delta_intensity', 0.0)

    # 1. Coherence Check
    # Determine the direction of flow and price
    flow_dir = 1 if mean_delta > 0 else -1 if mean_delta < 0 else 0
    price_dir = 1 if price_disp > 0 else -1 if price_disp < 0 else 0

    # Requirement: Flow and Price must both align with the predicted side
    is_coherent = (flow_dir == side_dir) and (price_dir == side_dir)
    
    # Also ensure flow is actually significant to avoid trading noise
    if not is_coherent or abs(mean_delta) < 0.005:
        return None, 0.0

    # 2. Dynamic EV Threshold
    # Base threshold is 0.03. As time runs out, we require more edge.
    # At 300s, threshold ~ 0.033. At 0s, threshold ~ 0.04.
    dynamic_threshold = 0.03 + (10.0 / (seconds_remaining + 1.0))

    if best_ev > dynamic_threshold:
        # 3. Scaled Bet Sizing
        # We target a 4% edge for a full bet, scaled by liquidity and intensity.
        # multiplier = (liquidity_quality) * (1 + log_intensity)
        # We cap the multiplier to prevent over-leveraging in extreme spikes.
        multiplier = min(1.5, lq * (1.0 + di))
        bet_fraction = (best_ev / 0.04) * multiplier

        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0