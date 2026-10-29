# Var 154 | BT: 0.696 | Live PnL: $-16.31 | Trades: 60
# Desc: Implements "Microstructure-Consensus Alignment (MCA)," which filters trades by ensuring the model's predicted direction aligns with the prevailing microstructural regime (Absorption vs. Trend) and sca
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 1.5191249999999998, 2.651597, -1.0, -1.0, 1.4562249999999999, 1.9791409999999998, -0.9504972696304321, -1.0, -0.6850976943969727, 1.28551, 1.890848, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -0.7141919136047363, -1.0, 0.21192830801010132, -1.0, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.020824909210205078, -1.0, -1.0, -0.6802398562431335, -1.0, -1.0, -1.0, -1.0, -0.8597651720046997, 0.13876599999999994, -1.0, 3.1194439999999997, -1.0, 1.0267908573150635, -0.9568802714347839, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.06756269931793213, 1.9791409999999998, 2.280573], "live_pnl": -16.313678741455078, "live_wins": 18, "live_total": 60, "age": 238, "rounds_underwater": 43, "abstain_rounds": 0, "backtest_score": 0.6956840093255003, "backtest_acc": 0.5684349631867095, "backtest_n": 1685}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding temporal 
    velocities to capture the rate of change in order flow and price.
    """
    df = df.copy()

    # 1. Temporal Velocities: Capture the momentum of the microstructural components
    # We group by market_slug to ensure diffs are only calculated within the same window.
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['funding_velocity'] = df.groupby('market_slug')['mean_funding'].diff().fillna(0)

    # 2. Interaction Terms: Already partially provided, but we add a key one:
    # Price-Flow Elasticity: How much price moves per unit of aggressive delta.
    # High elasticity = thin market; Low elasticity = thick/absorbed market.
    df['price_flow_elasticity'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 3. Spread-to-Price Ratio: Normalizing dispersion for scale-invariance
    df['dispersion_ratio'] = df['price_std'] / (df['binance_price'] + 1e-6)

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
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Consensus Alignment (MCA)".

    Logic:
    1. Identify Regime:
       - ABSORPTION: Delta and OBI have opposite signs (e.g., aggressive selling into heavy bids).
         The 'Wall' is the OBI. We only trade if the model predicts the OBI direction.
       - TREND: Delta and OBI have the same sign (e.g., aggressive buying into heavy bids).
         The 'Flow' is the driver. We only trade if the model predicts the Delta direction.
    
    2. Consensus Gating:
       - Use cross-exchange price dispersion (price_std) to scale conviction. 
         If exchanges disagree, the signal is unreliable.

    3. Execution:
       - Absorption: Low EV hurdle (0.025), High bet multiplier (1.3x).
       - Trend: Moderate EV hurdle (0.04), Reduced bet multiplier (0.8x).
       - Default: High EV hurdle (0.05), Minimal bet multiplier (0.5x).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract features safely
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    price_std = features_row.get('price_std', 0.0)
    binance_price = features_row.get('binance_price', 1.0)
    
    # 1. Determine Regime
    # Absorption: Flow is fighting the Book (High conviction reversal/wall)
    is_absorption = (mean_delta * mean_obi < 0) and (abs(mean_obi) > 0.05)
    # Trend: Flow is aligned with the Book (Momentum)
    is_trend = (mean_delta * mean_obi > 0) and (abs(mean_delta) > 0.001)

    threshold = 0.05
    multiplier = 0.5

    if is_absorption:
        threshold = 0.025
        multiplier = 1.3
        # In absorption, the 'Wall' (OBI) dictates the direction
        target_side = 'Up' if mean_obi > 0 else 'Down'
        if side != target_side:
            return None, 0.0
            
    elif is_trend:
        threshold = 0.04
        multiplier = 0.8
        # In trend, the 'Flow' (Delta) dictates the direction
        target_side = 'Up' if mean_delta > 0 else 'Down'
        if side != target_side:
            return None, 0.0
    
    # 2. Consensus & Time Scaling
    # consensus_score: 1.0 if exchanges are perfectly aligned, approaches 0 as dispersion grows
    consensus_score = 1.0 / (1.0 + (price_std / binance_price))
    
    # time_factor: reduce size as we approach the 5-min resolution cliff to avoid gamma noise
    time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)

    # 3. Final Decision
    if best_ev > threshold:
        # Scale bet fraction based on EV, regime multiplier, consensus, and time
        bet_fraction = (best_ev / 0.05) * multiplier * consensus_score * time_factor
        return side, min(1.0, max(0.01, bet_fraction))

    return None, 0.0