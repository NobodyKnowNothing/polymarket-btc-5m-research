# Var 190 | BT: 1.575 | Live PnL: $0.00 | Trades: 0
# Desc: Implements "Lead-Exchange Microstructure Pressure (LEMP)," which targets price catch-up by detecting when Binance's order-flow pressure (OBI $\times$ Delta) diverges significantly from the market mean
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 59, "rounds_underwater": 0, "abstain_rounds": 59, "backtest_score": 1.5754055486903702, "backtest_acc": 0.5649767940021421, "backtest_n": 23}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between Binance's microstructure pressure and the global average.
    """
    df = df.copy()

    # 1. Calculate Microstructure Pressure (OBI * Delta)
    # Pressure represents the 'force' of the order flow.
    df['binance_pressure'] = df['binance_obi'] * df['binance_delta']
    df['mean_pressure'] = df['mean_obi'] * df['mean_delta']
    
    # 2. Pressure Divergence (Lead-Exchange Signal)
    # Positive: Binance is pushing harder than the aggregate.
    # Negative: Binance is pulling back relative to the aggregate.
    df['pressure_lead'] = df['binance_pressure'] - df['mean_pressure']
    
    # 3. Pressure Acceleration
    # Captures the rate of change in the lead-exchange pressure.
    df['pressure_lead_accel'] = df.groupby('market_slug')['pressure_lead'].diff().fillna(0)

    # 4. Price-Lag Index
    # Measures how much price has moved relative to the aggressive flow (efficiency).
    # A low index suggests the price hasn't 'caught up' to the pressure yet.
    df['price_lag_index'] = df['price_vs_strike_pct'].abs() / (df['mean_delta'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
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
    Decide whether to trade using "Lead-Exchange Microstructure Pressure (LEMP)".

    Logic:
    1. Identify "Pressure Divergence":
       - We look for moments where Binance's pressure (OBI * Delta) is significantly 
         higher (or lower) than the cross-exchange mean.
    2. Identify "Price Lag":
       - We only enter if the price has not yet reacted to this pressure 
         (i.e., price_vs_strike_pct is small).
    3. Execution:
       - If Pressure is leading and Price is lagging, we bet on the 'catch-up' move.
       - Bet size is scaled by the magnitude of the pressure lead and the 
         time remaining in the window (to avoid late-window volatility).
    """
    # Exit if too little time remains to capture the catch-up move
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    pressure_lead = features_row.get('pressure_lead', 0.0)
    price_dist = features_row.get('price_vs_strike_pct', 0.0)

    side = None
    bet_fraction = 0.0

    # Thresholds
    # Pressure lead threshold (depends on delta scale, 20 is a robust starting point)
    pressure_threshold = 20.0
    # Price distance threshold (0.0015 = 15 basis points)
    max_price_lag = 0.0015
    # Minimum EV hurdle
    min_ev = 0.02

    # --- REGIME: LEAD-VENUE PRESSURE CATCH-UP ---
    
    # Case 1: Binance is pushing UP, but price hasn't jumped yet
    if pressure_lead > pressure_threshold and ev_up > min_ev:
        if abs(price_dist) < max_price_lag:
            side = 'Up'
            # Scale bet by pressure intensity and time remaining
            # Maximize bet when pressure is high and we are early in the window
            time_multiplier = seconds_remaining / 300.0
            pressure_multiplier = min(1.0, pressure_lead / 100.0)
            bet_fraction = pressure_multiplier * time_multiplier

    # Case 2: Binance is pushing DOWN, but price hasn't dropped yet
    elif pressure_lead < -pressure_threshold and ev_down > min_ev:
        if abs(price_dist) < max_price_lag:
            side = 'Down'
            time_multiplier = seconds_remaining / 300.0
            pressure_multiplier = min(1.0, abs(pressure_lead) / 100.0)
            bet_fraction = pressure_multiplier * time_multiplier

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0