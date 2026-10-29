# Var 88 | BT: -0.966 | Live PnL: $-4.72 | Trades: 61
# Desc: Implements "Predictive Flow-Skew Alignment (PFSA)," which filters trades by requiring the movement in Polymarket's implied probability (skew velocity) to be in strong congruence with aggressive spot e
# Stats: {"live_returns": [2.280573, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, 0.04885600000000001, 1.1846420000000002, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, -1.0, 2.651597, 0.04885600000000001, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, 0.15118699999999996, 0.15118699999999996, -1.0, -1.0], "live_pnl": -4.716667, "live_wins": 20, "live_total": 61, "age": 112, "rounds_underwater": 10, "abstain_rounds": 0, "backtest_score": -0.9656624972366603, "backtest_acc": 0.5646711012564671, "backtest_n": 2823}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the alignment between Polymarket price movement and spot flow.
    """
    df = df.copy()

    # 1. Polymarket Skew Dynamics
    # Skew = cost_up - cost_down. Positive skew means market leans Up.
    df['pm_skew'] = df['pm_cost_up'] - df['pm_cost_down']
    
    # Velocity of the skew (how fast is the market sentiment shifting?)
    df['pm_skew_vel'] = df.groupby('market_slug')['pm_skew'].diff()
    
    # Volatility of the skew (is the sentiment shifting smoothly or erratically?)
    df['pm_skew_vol'] = df.groupby('market_slug')['pm_skew'].transform(
        lambda x: x.rolling(window=5).std()
    )

    # 2. Flow-Skew Alignment Signal
    # A signal that captures if the change in Polymarket skew is supported by spot delta.
    # Positive = Skew and Delta are moving in the same direction.
    df['flow_support_signal'] = np.sign(df['pm_skew_vel']) * np.sign(df['mean_delta'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle NaNs from diff/rolling and infinite values
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to combat the 50/50 noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Predictive Flow-Skew Alignment (PFSA)".

    Logic:
    - Calculate raw EV for both sides.
    - Identify "Confirmed Moves": Where the direction of the Polymarket skew 
      velocity and the direction of the spot delta are both aligned with our side.
    - Identify "Divergent/Exhaustion Moves": Where spot flow and skew velocity 
      are fighting, or moving against our predicted side.
    - Apply a heavy multiplier to 'Confirmed' moves and a penalty to 'Divergent' moves.
    - Use time-decaying thresholds to avoid late-window volatility traps.
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

    # Extract engineered microstructure features
    # We use the signs of the velocity and delta to determine congruence
    s_vel = np.sign(features_row.get('pm_skew_vel', 0.0))
    s_delta = np.sign(features_row.get('mean_delta', 0.0))

    # 1. Determine Alignment Quality
    # A move is 'Confirmed' if both skew and delta move in the direction of our side.
    # A move is 'Divergent' if either is fighting our side.
    multiplier = 1.0
    if s_vel == side_dir and s_delta == side_dir:
        multiplier = 1.3  # High conviction: Sentiment and Flow are in sync
    elif s_vel == -side_dir or s_delta == -side_dir:
        multiplier = 0.5  # Low conviction: Sentiment and Flow are fighting
    # else: multiplier remains 1.0 (Neutral/No movement)

    # 2. Calculate Effective EV
    effective_ev = best_ev * multiplier

    # 3. Dynamic Thresholding
    # Require higher edge as the window closes to mitigate "noise-driven" terminal volatility.
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 150:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # 4. Dynamic Bet Sizing
        # Scale bet fraction based on the magnitude of the raw edge and the multiplier.
        # Target 1.0 bet when effective_ev is ~4%
        bet_fraction = (best_ev / 0.04) * multiplier
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0