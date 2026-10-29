# Var 97 | BT: -0.257 | Live PnL: $-6.91 | Trades: 61
# Desc: Implements "Cross-Venue Ignition & Dispersion Scaling (CVIDS)," which targets the precise moment aggressive order flow (delta acceleration) begins to pull exchange prices apart (dispersion velocity),
# Stats: {"live_returns": [1.5854279999999998, 0.10279029607772827, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7038174867630005, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, 1.3396370000000002, 0.09166908264160156, -1.0, 3.303756, 1.0943533182144165, 1.890848, -1.0, -1.0, -0.1, -0.8949649333953857, -0.1, -0.1, -1.0, 0.13876599999999994, 0.8534349799156189, -1.0, 4.550646, 3.1194439999999997, -1.0, 0.009182599999999999, -1.0, -1.0, -1.0, -0.8503841757774353, -1.0, -1.0, -0.1, 2.39541, 0.04971921443939209, 1.28551, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -0.1, -0.1, -1.0, -1.0, 1.9791409999999998, 0.11476700000000006, -0.1, -1.0, -0.1, -1.0, -1.0], "live_pnl": -6.90929651260376, "live_wins": 19, "live_total": 61, "age": 79, "rounds_underwater": 25, "abstain_rounds": 0, "backtest_score": -0.2574340994233605, "backtest_acc": 0.5581122329185255, "backtest_n": 3408}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the acceleration of order flow and the widening of cross-exchange dispersion.
    """
    df = df.copy()

    # 1. Flow Acceleration: Is the aggressive buying/selling picking up speed?
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    
    # 2. Dispersion Velocity: Is the price gap between exchanges widening or narrowing?
    # Widening dispersion during aggressive flow often signals a "lead" exchange ignition.
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 3. OBI Momentum: Is the order book imbalance trending?
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 4. Volatility-Adjusted Flow: Normalized delta
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 5. Price-Delta Cohesion: How well is price following the flow?
    df['price_momentum'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['cohesion'] = df['price_momentum'] / (df['mean_delta'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for detecting sudden microstructure shifts."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Venue Ignition & Dispersion Scaling (CVIDS)".

    Logic:
    - Ignition Up: Delta is accelerating, price spread is widening, and OBI is positive.
    - Ignition Down: Delta is decelerating, price spread is narrowing (or widening negatively), and OBI is negative.
    - This identifies the 'breakout' phase where one exchange leads the others.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    dv = features_row.get('delta_velocity', 0.0)
    sv = features_row.get('spread_velocity', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    cohesion = features_row.get('cohesion', 0.0)

    # 1. Detect Ignition Regime
    # Ignition occurs when aggressive flow (dv) and dispersion (sv) move in tandem with the book (obi)
    is_ignition_up = (dv > 0) and (sv > 0) and (obi > 0.1)
    is_ignition_down = (dv < 0) and (sv < 0) and (obi < -0.1)

    # 2. Determine Thresholds and Multipliers
    if is_ignition_up or is_ignition_down:
        # High conviction breakout; allow lower EV but scale by cohesion
        req_ev = 0.020
        # If price is already moving well with flow (high cohesion), increase bet
        multiplier = 0.8 + (min(cohesion, 0.5) / 0.5) * 0.4 
    else:
        # Standard regime; require higher edge to avoid noise
        req_ev = 0.035
        multiplier = 0.6

    # 3. Time-Gating
    # As the market closes, ignition signals become more volatile/unreliable
    if seconds_remaining < 45:
        req_ev *= 1.3
        multiplier *= 0.7

    # 4. Final Decision
    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% benchmark
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0