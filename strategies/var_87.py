# Var 87 | BT: 0.241 | Live PnL: $-2.78 | Trades: 61
# Desc: Implements "Absorption-Regime Adaptive Scaling (ARAS)," which uses the ratio of aggressive flow to price velocity to distinguish between momentum and absorption regimes, dynamically scaling conviction
# Stats: {"live_returns": [2.950496, 2.173394, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 2.950496, -1.0, -1.0, 0.04885600000000001, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, 2.651597, 1.5854279999999998, -1.0, -1.0, 0.06987699999999997, -1.0, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.09182599999999996, 2.073133, 0.09182599999999996, -1.0, 1.8077519999999998, 2.795066, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, -1.0], "live_pnl": -2.782909000000001, "live_wins": 22, "live_total": 61, "age": 178, "rounds_underwater": 4, "abstain_rounds": 0, "backtest_score": 0.24094185844328186, "backtest_acc": 0.548483341621084, "backtest_n": 2010}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between aggressive flow (delta) and price movement (velocity).
    """
    df = df.copy()

    # 1. Price Velocity: The change in price within the current market window
    # We group by market_slug to ensure we don't diff across different markets
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Absorption Ratio: How much delta is required to move the price?
    # High ratio = High absorption (large delta, little price move) -> Reversal potential
    # Low ratio = High efficiency (small delta, large price move) -> Momentum potential
    df['absorption_ratio'] = df['mean_delta'].abs() / (df['price_velocity'].abs() + 1e-6)

    # 3. Trend Direction: Sign alignment between flow and price
    # 1: Momentum (Delta and Price move together)
    # -1: Absorption/Reversal (Delta and Price move against each other)
    # 0: Neutral
    df['trend_direction'] = np.sign(df['mean_delta']) * np.sign(df['price_velocity'])

    # 4. Existing Microstructure Features (Interactions)
    df['delta_obi_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])
    df['price_dispersion_pct'] = df['price_std'] / (df['binance_price'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to prevent overfitting to microstructure noise."""
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
    Decide whether to trade using "Absorption-Regime Adaptive Scaling (ARAS)".

    Logic:
    - Identify the current regime: Momentum (trend_direction=1) or Absorption (trend_direction=-1).
    - If in a Momentum regime: High conviction if model predicts the trend; 
      low conviction if model predicts a reversal (scaled by 1/absorption_ratio).
    - If in an Absorption regime: High conviction if model predicts the reversal; 
      low conviction if model predicts the trend (scaled by absorption_ratio).
    - Use a dynamic EV threshold that tightens as the market window closes.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        model_side_val = 1
    else:
        best_ev = ev_down
        side = 'Down'
        model_side_val = -1

    # Extract engineered features
    trend_dir = features_row.get('trend_direction', 0)
    abs_ratio = features_row.get('absorption_ratio', 0)
    
    # 1. Regime-Adaptive Conviction Scaling
    # We want to scale the EV based on whether the model is "playing with" 
    # or "fighting" the current microstructure regime.
    if trend_dir == 0:
        # Neutral regime: no scaling
        regime_score = 0.5
    elif model_side_val == trend_dir:
        # Model predicts the trend (Momentum side)
        # Conviction is high when absorption is low (efficient movement)
        regime_score = 1.0 / (1.0 + abs_ratio)
    else:
        # Model predicts the reversal (Absorption side)
        # Conviction is high when absorption is high (heavy resistance)
        regime_score = abs_ratio / (1.0 + abs_ratio)

    # 2. Calculate Effective EV
    effective_ev = best_ev * regime_score

    # 3. Dynamic Thresholding
    # Tighten requirements as time runs out to avoid noise-driven trades.
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 150:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # Scale bet fraction: Target a 4% effective edge for a full 1.0 bet.
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0