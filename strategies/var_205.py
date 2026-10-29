# Var 205 | BT: -0.346 | Live PnL: $-13.91 | Trades: 60
# Desc: Implements "Predictive-Flow Convergence (PFC)," which validates model-predicted edges by requiring that real-time aggressive order flow (Delta) aligns with the predicted direction, preventing trades a
# Stats: {"live_returns": [-1.0, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, 1.890848, 1.5191249999999998, -1.0, -1.0, -1.0, 0.12663000000000002, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, 2.39541, 0.03256702423095703, 0.11476700000000006, -1.0, -1.0, 2.173394, -1.0, 0.12663000000000002, -1.0, -1.0, 1.6554160000000002, -0.9393080472946167, 1.3396370000000002, -1.0, 1.4562249999999999, -1.0, 3.505625, 0.1769179999999999, 0.1769179999999999, -1.0, -1.0, -1.0, -1.0, 3.727685, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.09182599999999996, 0.11476700000000006], "live_pnl": -13.90959644317627, "live_wins": 20, "live_total": 60, "age": 188, "rounds_underwater": 57, "abstain_rounds": 0, "backtest_score": -0.3462987179705126, "backtest_acc": 0.5769162995594713, "backtest_n": 1847}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between order flow efficiency and intent alignment.
    """
    df = df.copy()

    # 1. Delta-Volatility Efficiency
    # Measures how much aggressive flow is occurring relative to the current volatility.
    # High values suggest strong, meaningful moves; low values suggest noise or absorption.
    df['delta_vol_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)
    df['delta_vol_ratio'] = df['delta_vol_ratio'].clip(-10, 10)

    # 2. Intent-Action Synchronization
    # Does the order book imbalance (intent) match the aggressive delta (action)?
    df['obi_delta_sync'] = ((np.sign(df['mean_obi']) == np.sign(df['mean_delta'])) & (df['mean_delta'] != 0)).astype(float)

    # 3. Funding-Flow Alignment
    # Does the aggressive flow align with the perp funding sentiment?
    df['funding_flow_sync'] = ((np.sign(df['sum_funding']) == np.sign(df['mean_delta'])) & (df['mean_delta'] != 0)).astype(float)

    # 4. Price-Flow Efficiency
    # How much price moves per unit of delta (normalized by volatility).
    # High efficiency = breakout; Low efficiency = absorption.
    df['price_flow_efficiency'] = df['mean_delta'] / (df['price_vs_strike_pct'] + 1e-6)
    df['price_flow_efficiency'] = df['price_flow_efficiency'].clip(-10, 10)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on the most robust microstructure signals."""
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
    Decide whether to trade using "Predictive-Flow Convergence (PFC)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Flow Validation: 
       - A trade is only valid if the aggressive flow (Delta) is moving in the 
         SAME direction as the model's predicted edge.
       - If the model predicts 'Up' but Delta is negative, we abstain (market is leading the model).
    3. Dynamic Sizing:
       - Scale bet size by the magnitude of the EV.
       - Scale bet size by the 'delta_vol_ratio' (conviction in the flow).
       - Apply a time-gate to avoid late-window volatility spikes.
    """
    # Exit if too little time remains to capture meaningful movement
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Determine the best theoretical edge
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # Extract engineered features
    delta_vol_ratio = features_row.get('delta_vol_ratio', 0.0)
    
    # Minimum hurdle for any trade
    if best_ev < 0.03:
        return None, 0.0

    # --- FLOW CONVERGENCE GATE ---
    # If we want to go 'Up', delta_vol_ratio must be positive.
    # If we want to go 'Down', delta_vol_ratio must be negative.
    # This prevents "catching falling knives" where the model is right but the flow is wrong.
    is_flow_aligned = False
    if side == 'Up' and delta_vol_ratio > 0:
        is_flow_aligned = True
    elif side == 'Down' and delta_vol_ratio < 0:
        is_flow_aligned = True

    if not is_flow_aligned:
        return None, 0.0

    # --- DYNAMIC BET SIZING ---
    # Base size is driven by EV (normalized to 0.05)
    # Multiplier is driven by flow strength (delta_vol_ratio)
    base_size = (best_ev / 0.05)
    flow_multiplier = 1.0 + min(abs(delta_vol_ratio), 1.5)
    
    bet_fraction = min(1.0, base_size * flow_multiplier)

    # Safety floor for high-edge, high-flow trades
    if best_ev > 0.06 and abs(delta_vol_ratio) > 1.0:
        bet_fraction = max(bet_fraction, 0.5)

    return side, bet_fraction