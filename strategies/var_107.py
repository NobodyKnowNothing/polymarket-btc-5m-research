# Var 107 | BT: -0.218 | Live PnL: $-5.08 | Trades: 61
# Desc: Implements "Microstructure Pressure & Elasticity Scaling (MPES)", which differentiates between high-pressure absorption regimes (low price impact per unit of delta) and high-elasticity momentum regime
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 2.39541, 1.28551, -1.0, -1.0, 1.4562249999999999, -0.8369127511978149, -1.0, -1.0, -1.0, -0.6496977806091309, -1.0, -1.0, 1.9791409999999998, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.05862802267074585, 2.073133, 0.05862802267074585, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, -1.0, 1.7294070000000001, -1.0, 0.12663000000000002, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, 3.505625, 1.28551, 1.7294070000000001, 2.795066, 1.890848, 1.890848, -1.0, -1.0], "live_pnl": -5.080866813659668, "live_wins": 21, "live_total": 61, "age": 94, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.21828709487106782, "backtest_acc": 0.5350719424460432, "backtest_n": 3424}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between aggressive flow (delta) and price movement (elasticity).
    """
    df = df.copy()

    # 1. Price Velocity: Change in price within the market window
    # We group by market_slug to ensure diffs don't leak between different 5-min windows
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Pressure Index: Measures "Absorption". 
    # High pressure = High delta but low price movement (hitting a wall).
    # Low pressure = High delta and high price movement (efficient momentum).
    # We use log1p to stabilize the scale of this ratio.
    df['pressure_index'] = np.log1p(abs(df['mean_delta']) / (abs(df['price_velocity']) + 1e-6))

    # 3. Microstructure Congruence: Do the aggressive flow and the book imbalance agree?
    # 1.0 = Agreement (Buying into bids), -1.0 = Disagreement (Buying into asks/absorption)
    df['congruence'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 4. Elasticity: Price movement per unit of delta
    df['elasticity'] = abs(df['price_velocity']) / (abs(df['mean_delta']) + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Pressure & Elasticity Scaling (MPES)".

    Logic:
    - Regime 1: Absorption (High Pressure). Price is stalling despite aggressive flow.
      This often precedes a violent breakout or a reversal. We lower the EV hurdle 
      but increase the bet multiplier to capture the "coiled spring" effect.
    - Regime 2: Momentum (Low Pressure). Price moves easily with flow.
      We require a higher EV hurdle to avoid chasing exhausted moves.
    - Congruence Penalty: If Delta and OBI disagree (congruence < 0), the signal 
      is likely high-friction absorption; we significantly raise the EV hurdle.
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
    pressure = features_row.get('pressure_index', 0.0)
    congruence = features_row.get('congruence', 0.0)

    # 1. Identify Regime and set base thresholds
    # pressure_index > 1.5 is a heuristic for high-absorption/low-elasticity
    if pressure > 1.5:
        # Absorption Regime: High potential for explosive movement
        req_ev = 0.020
        multiplier = 1.3
    else:
        # Momentum Regime: Standard trend following
        req_ev = 0.035
        multiplier = 0.7

    # 2. Congruence Penalty: If flow and book are fighting, increase the hurdle
    if congruence < 0:
        req_ev += 0.025

    # 3. Time-Gating: Tighten requirements as the window closes
    if seconds_remaining < 60:
        req_ev *= 1.5
        multiplier *= 0.8

    # 4. Execution
    if best_ev > req_ev:
        # Scale bet size based on edge relative to a 4% target
        # We cap the multiplier to ensure we don't over-leverage on single trades
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0