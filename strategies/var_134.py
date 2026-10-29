# Var 134 | BT: -0.708 | Live PnL: $-6.13 | Trades: 61
# Desc: Implements "Microstructure Friction & Efficiency Scaling (MFES)," which distinguishes between "clean" momentum (high price movement per unit of delta) and "absorption" (high delta with minimal price m
# Stats: {"live_returns": [-1.0, -1.0, -0.6024172902107239, 1.055962085723877, -0.5137453079223633, -0.517162024974823, -1.0, -0.6220076084136963, 0.09070539474487305, -1.0, -0.57187819480896, 1.7294070000000001, -1.0, -0.5585638880729675, -0.43194499611854553, -0.6855450868606567, -1.0, 0.9151471257209778, -1.0, -0.23977522552013397, -1.0, 2.39541, 2.280573, 1.9791409999999998, -0.29263773560523987, 1.890848, -1.0, -1.0, -1.0, 0.552037239074707, 0.7847302556037903, 1.7423069477081299, 1.7294070000000001, -0.7130988836288452, 3.727685, -1.0, -0.9573664665222168, 0.9552936553955078, -0.5310289263725281, -0.6220008730888367, 0.16389999999999993, 2.39541, -1.0, -0.3547862470149994, -0.37788474559783936, -0.4706072509288788, -1.0, -1.0, -1.0, -0.6024172902107239, -1.0, -0.47543609142303467, -1.0, 0.10316799999999993, 1.7294070000000001, -1.0, 0.13876599999999994, -1.0, -0.4271632134914398, -1.0, -0.9214422106742859], "live_pnl": -6.129604339599609, "live_wins": 19, "live_total": 61, "age": 98, "rounds_underwater": 13, "abstain_rounds": 0, "backtest_score": -0.7076751306279305, "backtest_acc": 0.5522777443307245, "backtest_n": 3347}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement relative to aggressive flow.
    """
    df = df.copy()

    # 1. Normalized Price Velocity: Rate of price change relative to the remaining window.
    # We normalize by (seconds_remaining / 300) so the velocity is comparable 
    # across different stages of the 5-minute window.
    time_factor = (df['seconds_remaining'] / 300.0) + 1e-6
    df['price_velocity_norm'] = (df['binance_price'] - df['strike_price']) / time_factor

    # 2. Delta Efficiency: How much price moves per unit of aggressive delta.
    # High efficiency = Clean trend. Low efficiency = Absorption/Battle.
    # We use the absolute mean_delta to measure the 'effort' being exerted.
    df['delta_efficiency'] = df['price_velocity_norm'] / (df['mean_delta'].abs() + 1e-6)

    # 3. Binance Price Lead: Distance of Binance from the cross-exchange average.
    df['binance_price_lead'] = df['binance_price'] - (df['coinbase_price'] + df['kraken_price']) / 2

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters optimized for noisy microstructure signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Friction & Efficiency Scaling (MFES)".

    Logic:
    - Calculate raw EV for both sides.
    - Efficiency Metric: Uses 'delta_efficiency' to determine if the market move is 'clean' or 'fighting'.
    - High Efficiency (> 0.05): The move is a clean breakout. If Binance lead aligns with 
      the side, amplify EV (1.5x). If it contradicts, penalize (0.6x).
    - Low Efficiency (near 0): The move is an absorption battle. Significantly penalize EV (0.3x).
    - Negative Efficiency: Price is moving against the flow (potential exhaustion). Penalize (0.5x).
    - Bet Sizing: Scales bet fraction based on the resulting 'effective_ev'.
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
    eff = features_row.get('delta_efficiency', 0.0)
    lead = features_row.get('binance_price_lead', 0.0)

    # 1. Determine Multiplier based on Microstructure Quality
    multiplier = 1.0
    
    if eff > 0.05:
        # High Efficiency: Clean breakout/trend
        # If we are going Up, we want Binance leading bullishly.
        if (side == 'Up' and lead > 0) or (side == 'Down' and lead < 0):
            multiplier = 1.5
        else:
            multiplier = 0.6
    elif -0.05 < eff < 0.05:
        # Low Efficiency: Absorption/Friction (High Delta, Low Price Move)
        multiplier = 0.3
    elif eff <= -0.05:
        # Negative Efficiency: Price moving against aggressive flow (Exhaustion/Reversal)
        multiplier = 0.5

    # 2. Calculate Effective EV
    effective_ev = best_ev * multiplier

    # 3. Dynamic Thresholding and Bet Sizing
    # We require a higher effective edge to compensate for the scaling.
    base_threshold = 0.03
    if seconds_remaining < 45:
        base_threshold = 0.045

    if effective_ev > base_threshold:
        # Scale bet fraction: Target 4% effective edge for a full bet.
        bet_fraction = (effective_ev / 0.04) * (multiplier if multiplier < 1.0 else 1.0)
        # Ensure we don't over-leverage on amplified bets
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0