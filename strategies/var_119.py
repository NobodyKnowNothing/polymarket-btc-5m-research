# Var 119 | BT: -1.533 | Live PnL: $-2.85 | Trades: 61
# Desc: Implements "Kinetic-Flow Dislocation (KFD) Gating," which targets the high-alpha window where price velocity and aggressive delta acceleration are synchronized but the order book (OBI) and Polymarket
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 6.238865, 3.727685, -1.0, -1.0, -1.0, 0.500645, -1.0, -1.0, 2.950496, -1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, 2.651597, -1.0, 0.09182599999999996, 0.09182599999999996, 2.950496, -1.0, 1.7645650000000002, -1.0, 0.293439, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, 4.550646, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -2.8468989999999987, "live_wins": 18, "live_total": 61, "age": 161, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": -1.5326810283044325, "backtest_acc": 0.5566587524282323, "backtest_n": 3352}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding second-order 
    velocity and acceleration metrics to capture momentum ignition.
    """
    df = df.copy()

    # Ensure we are working with sorted time within each market to compute diffs correctly
    df = df.sort_values(['market_slug', 'timestamp'])

    # 1. Velocity Metrics (Rate of change of price and flow)
    # Captures how fast the market is moving in the current window
    group = df.groupby('market_slug')
    df['price_velocity'] = group['binance_price'].diff().fillna(0)
    df['delta_velocity'] = group['sum_delta'].diff().fillna(0)
    df['obi_velocity'] = group['mean_obi'].diff().fillna(0)

    # 2. Acceleration Metrics (Rate of change of velocity)
    # Captures the "ignition" phase of a move
    df['price_accel'] = df['price_velocity'].groupby(df['market_slug']).diff().fillna(0)
    df['delta_accel'] = df['delta_velocity'].groupby(df['market_slug']).diff().fillna(0)

    # 3. Dislocation Metrics
    # Relationship between price movement and order flow strength
    df['flow_price_efficiency'] = df['price_velocity'] / (df['sum_delta'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for detecting non-linear momentum bursts."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Kinetic-Flow Dislocation (KFD) Gating".

    Logic:
    - Identifies "Ignition": When price velocity and delta velocity are both accelerating 
      in the same direction, suggesting a breakout.
    - Identifies "Friction": When price moves but delta or OBI lags or moves oppositely, 
      suggesting absorption or a fake-out.
    - Scales bet size based on the congruence of these kinetic signals.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered kinetic features
    p_vel = features_row.get('price_velocity', 0.0)
    d_vel = features_row.get('delta_velocity', 0.0)
    o_vel = features_row.get('obi_velocity', 0.0)
    
    # 1. Calculate Congruence Score
    # A score > 1.0 means signals are aligned (Ignition)
    # A score < 1.0 means signals are conflicting (Friction/Absorption)
    congruence = 1.0

    if side == 'Up':
        # For an 'Up' bet, we want positive price velocity and positive delta velocity
        if p_vel > 0 and d_vel > 0:
            congruence = 1.5  # Ignition
        elif p_vel > 0 and d_vel < 0:
            congruence = 0.5  # Price moving up but aggressive selling (Absorption)
        elif p_vel < 0:
            congruence = 0.4  # Price moving against our side
            
    elif side == 'Down':
        # For a 'Down' bet, we want negative price velocity and negative delta velocity
        if p_vel < 0 and d_vel < 0:
            congruence = 1.5  # Ignition
        elif p_vel < 0 and d_vel > 0:
            congruence = 0.5  # Price moving down but aggressive buying (Absorption)
        elif p_vel > 0:
            congruence = 0.4  # Price moving against our side

    # 2. Apply Effective EV Scaling
    effective_ev = best_ev * congruence

    # 3. Time-Gated Hurdle
    # As time runs out, we require higher absolute EV to compensate for volatility noise
    # Threshold scales from 0.02 at 5m to 0.07 at 0s
    threshold = 0.02 + (0.05 * (1.0 - (min(seconds_remaining, 300.0) / 300.0)))

    if effective_ev > threshold:
        # 4. Dynamic Sizing
        # Scale bet size by the effective edge, capped at 1.0
        # We use a base denominator of 0.04 (4% edge)
        bet_fraction = (effective_ev / 0.04) * (0.8 + 0.4 * congruence)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0