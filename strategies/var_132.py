# Var 132 | BT: -1.055 | Live PnL: $-3.83 | Trades: 61
# Desc: Implements "Strike-Proximity Impulse & Tension Gating (SPITG)," which scales bet size based on the convergence of price-to-strike proximity, order-flow impulse, and cross-exchange price tension.
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, -1.0, -1.0, -1.0, 1.1375739999999999, 2.39541, 1.28551, 1.049452, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, 3.727685, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0], "live_pnl": -3.8253359999999987, "live_wins": 21, "live_total": 61, "age": 82, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -1.0553673322798527, "backtest_acc": 0.5561831753076457, "backtest_n": 4093}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding temporal 
    derivatives (rates of change) and strike proximity to capture market energy.
    """
    df = df.copy()

    # 1. Temporal Derivatives (Rates of Change)
    # We group by market_slug to ensure diffs are only calculated within the same window.
    group = df.groupby('market_slug')
    df['delta_rate'] = group['sum_delta'].diff().fillna(0)
    df['obi_rate'] = group['mean_obi'].diff().fillna(0)
    df['spread_rate'] = group['price_spread'].diff().fillna(0)

    # 2. Strike Proximity
    # Measures how close the current price is to the strike. 
    # High value = near strike (high sensitivity); Low value = far from strike.
    df['proximity'] = 1.0 / (1.0 + abs(df['price_vs_strike_pct']) * 100)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return extremely regularized hyperparameters to handle noisy temporal features."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 50.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Strike-Proximity Impulse & Tension Gating (SPITG)".

    Logic:
    - Calculates raw EV for both sides.
    - Identifies "Market Energy" via three components:
        1. Proximity: Is the price near the strike (where moves matter most)?
        2. Impulse: Is there a sudden burst in Delta or OBI?
        3. Tension: Is there high dispersion (spread) between exchanges?
    - Instead of scaling the EV threshold (which can lead to bad trades), 
      we use these components to scale the 'bet_fraction' (conviction).
    - Implements a "Time-Decay Safety": Increases the EV requirement as the 
      window closes to avoid gambling on last-second noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 1. Dynamic Thresholding
    # As time runs out, we require a higher edge to compensate for volatility spikes.
    base_threshold = 0.035
    if seconds_remaining < 45:
        base_threshold = 0.05
    
    if best_ev < base_threshold:
        return None, 0.0

    # 2. Extract Engineered Features
    proximity = features_row.get('proximity', 0.5)
    # Impulse is the combined magnitude of delta and OBI movement
    impulse = min(abs(features_row.get('delta_rate', 0.0)) + abs(features_row.get('obi_rate', 0.0)), 2.0)
    # Tension is the current cross-exchange price spread
    tension = features_row.get('price_spread', 0.0)

    # 3. Calculate Conviction Multiplier
    # We scale the bet size based on how much "energy" is in the market.
    # Proximity (up to 0.5 boost), Impulse (up to 0.6 boost), Tension (up to 0.4 boost)
    conviction_multiplier = 1.0 + (proximity * 0.5) + (impulse * 0.3) + (min(tension, 0.05) * 8.0)

    # 4. Final Decision
    # Target a 4% edge for a full bet, scaled by our conviction multiplier.
    bet_fraction = (best_ev / 0.04) * conviction_multiplier
    
    return side, min(1.0, max(0.1, bet_fraction))