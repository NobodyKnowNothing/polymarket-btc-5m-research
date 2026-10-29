# Var 167 | BT: -0.143 | Live PnL: $-15.10 | Trades: 60
# Desc: Implements "Triple-Threat Acceleration Alignment (TTAA)," which differentiates between sustainable momentum and exhaustion by requiring alignment between aggressive flow (Delta), order book intent (OB
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, 0.10999464988708496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, 1.1985992193222046, -1.0, -1.0, 1.4562249999999999, -0.8518946170806885, 4.245841, -1.0, 2.518758, -1.0, -1.0, 0.03171950578689575, -0.9437969326972961, 1.4562249999999999, -1.0, 1.233902, -1.0, -1.0, -1.0, -1.0, 0.02597188949584961, -1.0, -0.4362669587135315, -1.0, -1.0, -1.0, 9.427877, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.12052005529403687, -1.0, 0.05925399999999992, 0.05925399999999992, -0.8448377251625061, -1.0, -1.0], "live_pnl": -15.097614288330078, "live_wins": 16, "live_total": 60, "age": 135, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": -0.14326146509225426, "backtest_acc": 0.5436116793751162, "backtest_n": 2540}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically 
    engineering second-order microstructure signals (acceleration).
    """
    df = df.copy()

    # 1. Microstructure Acceleration: Capture the rate of change in flow and intent
    # We group by market_slug to ensure diffs are only calculated within the same window
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_accel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 2. Flow-Price Coherence: Does the price velocity align with the delta?
    # price_vs_strike_pct is already provided in the pre-parsed features
    df['flow_price_coherence'] = df['mean_delta'] * (df['price_vs_strike_pct'] + 1e-6)

    # 3. Standard interaction features
    df['obi_delta_interaction'] = df['mean_obi'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """
    Return hyperparameters tuned for capturing interaction between 
    level-1 (delta) and level-2 (acceleration) features.
    """
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Triple-Threat Acceleration Alignment (TTAA)".

    Logic:
    1. Identify Signal Quality:
       - TRIPLE THREAT (High Conviction): Delta, OBI, and Delta-Acceleration all share the same sign.
         (e.g., buying, bids are stacking, and buying is speeding up).
       - DUAL THREAT (Medium Conviction): Delta and OBI share the same sign, but acceleration is fading.
       - EXHAUSTION (No Trade): Delta and OBI have different signs, or Delta is moving against its own acceleration.
    2. Execution:
       - Scale conviction: Triple Threat = 1.5x, Dual Threat = 1.0x.
       - Apply a time-decay factor to reduce exposure in the final 30 seconds of high-volatility windows.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    delta_accel = features_row.get('delta_accel', 0.0)

    # Determine signs
    delta_sign = 1 if mean_delta > 0 else (-1 if mean_delta < 0 else 0)
    obi_sign = 1 if mean_obi > 0 else (-1 if mean_obi < 0 else 0)
    accel_sign = 1 if delta_accel > 0 else (-1 if delta_accel < 0 else 0)

    # 1. Regime Identification
    conviction = 0.0 

    if delta_sign != 0:
        # Check if OBI and Delta are aligned (Dual Threat)
        if obi_sign == delta_sign:
            # Check if Acceleration is also aligned (Triple Threat)
            if accel_sign == delta_sign:
                conviction = 1.5
            else:
                conviction = 1.0
        else:
            # Divergence/Absorption: Delta and OBI fighting
            conviction = 0.0

    # 2. Adjusted EV Thresholding
    # We require a higher effective EV to compensate for the risk of noise
    effective_ev = best_ev * conviction

    if effective_ev > 0.02:
        # 3. Dynamic Bet Sizing
        # Target edge of 5% (0.05) for full sizing
        # Add a small time-safety factor for the final 30 seconds
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = (best_ev / 0.05) * conviction * time_factor

        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0