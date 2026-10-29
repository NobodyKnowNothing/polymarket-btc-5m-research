# Var 94 | BT: -0.002 | Live PnL: $-1.43 | Trades: 70
# Desc: Implements "Flow-Volatility Coherence (FVC)," which scales conviction based on the alignment of order flow and book imbalance, normalized by the current volatility regime to avoid over-betting during
# Stats: {"live_returns": [-1.0, 3.9731249999999996, 1.28551, 1.28551, -1.0, -1.0, -1.0, 2.280573, 0.15118699999999996, 0.04885600000000001, -1.0, -1.0, 4.550646, -1.0, 6.238865, 3.727685, 0.08073200000000003, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, 4.550646, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.28551, 2.795066, 3.727685, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.432418999999996, "live_wins": 20, "live_total": 70, "age": 251, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.002263858341955527, "backtest_acc": 0.5783333333333334, "backtest_n": 1644}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between aggressive flow and order book pressure.
    """
    df = df.copy()

    # 1. Flow-Volatility Coherence: 
    # Measures how much 'pressure' (Delta * OBI) is being applied per unit of volatility.
    # High positive = aggressive buying in a thick book.
    # High negative = aggressive selling in a thick book.
    # Low = noise or high-volatility chaos.
    df['flow_coherence'] = (df['mean_delta'] * df['mean_obi']) / (df['mean_volatility'] + 1e-6)

    # 2. Price Stability:
    # Inverse of price dispersion; higher means exchanges are in tight agreement.
    df['price_stability'] = 1.0 / (df['price_std'] + 1e-6)

    # 3. Delta Intensity:
    # Aggressive flow relative to the current volatility regime.
    df['delta_intensity'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

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
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Flow-Volatility Coherence (FVC) Scaling".

    Logic:
    - Calculate raw EV for both sides.
    - Identify the best side based on EV.
    - Check for 'Coherence': The model's preferred side must align with the 
      sign of the 'flow_coherence' (Delta * OBI / Vol).
    - If the signal is coherent (e.g., model wants 'Up' and flow/OBI is positive),
      scale the bet fraction using an arctangent function of the coherence magnitude.
      This allows for higher conviction in 'clean' moves and lower conviction in 'noisy' ones.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Threshold for entry
    base_threshold = 0.03
    
    # Adjust threshold based on time remaining (tighten as window closes)
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 150:
        base_threshold = 0.04

    if best_ev < base_threshold:
        return None, 0.0

    # Extract coherence
    coherence = features_row.get('flow_coherence', 0.0)

    # 1. Coherence Check: 
    # If we are betting 'Up', coherence must be positive.
    # If we are betting 'Down', coherence must be negative.
    is_coherent = False
    if side == 'Up' and coherence > 0:
        is_coherent = True
    elif side == 'Down' and coherence < 0:
        is_coherent = True

    if not is_coherent:
        return None, 0.0

    # 2. Dynamic Bet Sizing:
    # Scale bet based on how much the EV exceeds the threshold, 
    # amplified by the magnitude of the coherence (using atan to cap the multiplier).
    # Target: A coherence of ~1.0 and EV of 0.04 should yield a near-full bet.
    
    # base_scale: how much of the 0.04 target edge we have
    base_scale = best_ev / 0.04
    
    # coherence_multiplier: ranges from 1.0 to ~2.57 (1 + pi/2)
    coherence_multiplier = 1.0 + math.atan(abs(coherence))
    
    bet_fraction = base_scale * coherence_multiplier

    return side, min(1.0, max(0.1, bet_fraction))