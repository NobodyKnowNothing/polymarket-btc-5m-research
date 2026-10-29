# Var 176 | BT: 0.208 | Live PnL: $-21.73 | Trades: 60
# Desc: Implements "Momentum Acceleration & Exhaustion Gating (MAEG)," which uses the second derivative (acceleration) of order flow and imbalance to distinguish between sustainable breakouts and fading momen
# Stats: {"live_returns": [0.05410122871398926, -1.0, -1.0, -0.6201125979423523, -1.0, -1.0, 1.4562249999999999, -1.0, 1.233902, 2.280573, -1.0, 0.11476700000000006, -1.0, 0.09211975336074829, 1.0898240000000001, -1.0, -1.0, 0.03317636251449585, 0.10128748416900635, -1.0, 1.039311408996582, 0.03317636251449585, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, 1.5191249999999998, -0.5516815781593323, -1.0, -1.0, -1.0, -1.0, -0.43825864791870117, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.6763048768043518, -1.0, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, 1.6554160000000002, -0.6404869556427002, -1.0, 0.12663000000000002, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, 1.3964720000000002], "live_pnl": -21.726974487304688, "live_wins": 18, "live_total": 60, "age": 78, "rounds_underwater": 59, "abstain_rounds": 0, "backtest_score": 0.20778597546639432, "backtest_acc": 0.5629114850036576, "backtest_n": 4567}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the first and second derivatives (velocity and acceleration) of 
    order flow (Delta) and order book imbalance (OBI).
    """
    df = df.copy()

    # 1. Delta Dynamics (Aggressive Flow)
    # Velocity: How much the net flow is changing
    df['delta_velocity'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)
    # Acceleration: Is the flow speeding up or slowing down?
    df['delta_accel'] = df.groupby('market_slug')['delta_velocity'].diff().fillna(0)

    # 2. OBI Dynamics (Order Book Intent)
    # Velocity: Change in mean imbalance
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    # Acceleration: Change in the rate of imbalance change
    df['obi_accel'] = df.groupby('market_slug')['obi_velocity'].diff().fillna(0)

    # 3. Price Dynamics
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['price_accel'] = df.groupby('market_slug')['price_velocity'].diff().fillna(0)

    # 4. Interaction: Delta-to-Price Efficiency
    # Measures how much 'push' is required to move price.
    df['flow_impact_ratio'] = df['sum_delta'].abs() / (df['price_velocity'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for capturing momentum-shift signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Momentum Acceleration & Exhaustion Gating (MAEG)".

    Logic:
    1. Calculate EV for both sides.
    2. Detect "Exhaustion": If velocity and acceleration have opposite signs 
       (e.g., buying is still happening but the rate of buying is slowing down), 
       the move is likely losing steam.
    3. Detect "Breakout": If velocity and acceleration have the same sign, 
       momentum is accelerating.
    4. Apply a Volatility Hurdle: Increase the required EV if the market is in a 
       high-volatility regime to avoid noise.
    5. Time-Gating: Scale bet size down as the window approaches zero.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Determine the best side
    if ev_up > ev_down and ev_up > 0.02:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up and ev_down > 0.02:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    d_vel = features_row.get('delta_velocity', 0)
    d_acc = features_row.get('delta_accel', 0)
    o_vel = features_row.get('obi_velocity', 0)
    o_acc = features_row.get('obi_accel', 0)
    
    # Check volatility regime (handle both string and float if available)
    vol_regime = features_row.get('volatility_regime', 'low')
    is_high_vol = (vol_regime == 'high') or (features_row.get('mean_volatility', 0) > 0.005)

    # 1. Exhaustion Multiplier
    # If velocity and acceleration have different signs, momentum is decelerating.
    exhaustion_multiplier = 1.0
    
    # Check Delta Exhaustion (Aggressive flow slowing down)
    if d_vel != 0 and d_acc != 0:
        if (d_vel > 0 and d_acc < 0) or (d_vel < 0 and d_acc > 0):
            exhaustion_multiplier *= 0.6  # Significant reduction in conviction
            
    # Check OBI Exhaustion (Order book intent fading)
    if o_vel != 0 and o_acc != 0:
        if (o_vel > 0 and o_acc < 0) or (o_vel < 0 and o_acc > 0):
            exhaustion_multiplier *= 0.8

    # 2. Volatility Hurdle
    # In high vol, we require a higher EV to compensate for noise.
    required_ev_threshold = 0.035 if is_high_vol else 0.02
    if best_ev < required_ev_threshold:
        return None, 0.0

    # 3. Time Decay Factor
    # Reduce exposure in the final 30 seconds to avoid "last-second" volatility.
    time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)

    # 4. Final Sizing
    # Base bet is scaled by (EV / Target_EV) * Exhaustion * Time
    target_ev = 0.04
    bet_fraction = (best_ev / target_ev) * exhaustion_multiplier * time_factor

    # Clamp bet_fraction between 0.05 and 1.0
    return side, min(1.0, max(0.05, bet_fraction))