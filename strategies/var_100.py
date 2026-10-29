# Var 100 | BT: 0.458 | Live PnL: $-8.11 | Trades: 61
# Desc: Implements "Microstructure Exhaustion & Impact Efficiency Gating (MEIEG)," which distinguishes between momentum breakouts, delta exhaustion, and absorption traps to dynamically adjust EV thresholds an
# Stats: {"live_returns": [-1.0, 0.14798861742019653, -1.0, -1.0, -0.49070820212364197, -1.0, -1.0, -0.3121257424354553, 3.1194439999999997, 0.116091787815094, -1.0, 2.795066, -1.0, -1.0, 1.3964720000000002, -0.5514612793922424, -1.0, 6.118911266326904, -1.0, -1.0, -1.0, -1.0, 0.03888982534408569, 2.39541, -1.0, -0.3644927442073822, -1.0, 0.8376032710075378, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -0.4964342713356018, -1.0, -1.0, 2.39541, 0.03869050741195679, 1.208299994468689, -1.0, -1.0, 1.2798992395401, -1.0, -1.0, -1.0, -0.8700867891311646, -1.0, -1.0, 1.9791409999999998, 0.08758336305618286, -0.8789191246032715, -1.0, -1.0, -1.0, -0.8945642113685608, -1.0, -0.5215595960617065, -1.0, -0.4973919987678528, -0.8589012026786804], "live_pnl": -8.111654281616211, "live_wins": 17, "live_total": 61, "age": 95, "rounds_underwater": 10, "abstain_rounds": 0, "backtest_score": 0.45817530615741064, "backtest_acc": 0.5625994092251762, "backtest_n": 2955}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    delta acceleration, price-impact efficiency, and volatility-adjusted spreads.
    """
    df = df.copy()

    # 1. Price and Delta Dynamics
    # Calculate velocity (1st derivative) and acceleration (2nd derivative)
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Impact Efficiency: How much price moves per unit of aggressive delta.
    # High efficiency = liquidity gap/breakout. Low efficiency = absorption.
    df['impact_efficiency'] = df['price_velocity'] / (df['mean_delta'].abs() + 1e-6)

    # 3. Market Stability: Ratio of price dispersion to volatility.
    # High ratio suggests the spread is widening faster than the price is moving (instability).
    df['vol_spread_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters optimized for capturing subtle microstructure shifts."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Exhaustion & Impact Efficiency Gating (MEIEG)".

    Logic:
    - Breakout: Price and Delta are aligned AND Delta is accelerating. (Low EV threshold)
    - Exhaustion: Price and Delta are aligned BUT Delta is decelerating. (Medium EV threshold)
    - Absorption: Price and Delta are opposing. (High EV threshold)
    - Unstable: High volatility-to-spread ratio. (Penalty applied)
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
    p_vel = features_row.get('price_velocity', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    d_vel = features_row.get('delta_velocity', 0.0)
    vol_spread_ratio = features_row.get('vol_spread_ratio', 0.0)

    # 1. Identify Microstructure States
    # Breakout: Delta and Price move together and Delta is increasing
    is_breakout = (delta * p_vel > 0) and (delta * d_vel > 0)
    
    # Exhaustion: Delta and Price move together but Delta is slowing down
    is_exhaustion = (delta * p_vel > 0) and (delta * d_vel < 0)
    
    # Absorption: Price moves against the aggressive delta
    is_absorption = (delta * p_vel < 0)

    # 2. Determine Required EV Threshold
    if is_breakout:
        req_ev = 0.025  # Aggressive entry on momentum
    elif is_exhaustion:
        req_ev = 0.045  # Wait for more conviction as flow slows
    elif is_absorption:
        req_ev = 0.060  # High risk of being 'run over'
    else:
        req_ev = 0.035  # Standard neutral regime

    # 3. Apply Stability Penalty
    # If the spread is widening relative to volatility, increase required edge
    if vol_spread_ratio > 2.5:
        req_ev *= 1.4

    # 4. Execution and Sizing
    # Time decay: Reduce exposure as the window closes to avoid end-of-window noise
    time_decay = min(1.0, seconds_remaining / 300.0)

    if best_ev > req_ev:
        # Scale bet size: Target 4% EV as the baseline for 1.0 bet_fraction
        # We use a non-linear scaling to reward higher conviction
        bet_fraction = (best_ev / 0.04) * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0