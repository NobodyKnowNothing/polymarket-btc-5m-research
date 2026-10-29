# Var 192 | BT: 0.155 | Live PnL: $-10.73 | Trades: 21
# Desc: Implements "Microstructure-Polymarket Pricing Gap (MPPG)," which filters model signals by requiring that the Polymarket price is significantly lagging the immediate order book pressure (OBI + Delta-im
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -0.45468422155586496, -1.0, -1.0, -1.0, -1.0, -1.0, 0.9068706622852378, -0.6050124960206543, 1.28551, -1.0, -0.7479571884134667], "live_pnl": -10.725955243704746, "live_wins": 4, "live_total": 21, "age": 529, "rounds_underwater": 21, "abstain_rounds": 58, "backtest_score": 0.15460796488149742, "backtest_acc": 0.5513301196214961, "backtest_n": 29}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the discrepancy between exchange microstructure pressure and Polymarket pricing.
    """
    df = df.copy()

    # 1. Normalize Delta by volatility to get 'Impact-Adjusted Delta'
    # This tells us how much aggressive flow is actually moving the needle.
    df['norm_delta'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 2. Calculate Microstructure Pressure Index (MPI)
    # Combines OBI (intent) and Normalized Delta (action) into a single [-1, 1] metric.
    # We use tanh to squash the delta component to ensure it doesn't overwhelm OBI.
    df['micro_pressure'] = (df['mean_obi'] + np.tanh(df['norm_delta'])) / 2.0

    # 3. Calculate the Pricing Gap (The core alpha feature)
    # Maps micro_pressure from [-1, 1] to [0, 1] to compare directly with pm_cost_up.
    df['micro_prob'] = (df['micro_pressure'] + 1.0) / 2.0
    df['pricing_gap'] = df['micro_prob'] - df['pm_cost_up']

    # Drop metadata and target
    # Note: seconds_remaining is metadata but used in strategy_filter, not X.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Polymarket Pricing Gap (MPPG)".

    Logic:
    The model provides a directional probability. However, we only execute if 
    the Polymarket price is 'mispriced' relative to the immediate order book 
    pressure (micro_prob). 
    
    If the model says 'Up' and the order book is screaming 'Up' (high micro_prob), 
    but the Polymarket cost is still low (high pricing_gap), we have a high-conviction 
    mispricing opportunity.
    """
    # Exit if too little time remains
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    # micro_prob is the 'implied probability' from the order book
    # pricing_gap is (micro_prob - cost_up)
    micro_prob = features_row.get('micro_prob', 0.5)
    pricing_gap = features_row.get('pricing_gap', 0.0)

    side = None
    bet_fraction = 0.0

    # --- UP SIDE LOGIC ---
    # We want: Model is bullish (prob_up > 0.5) AND 
    # there is positive EV AND 
    # the order book is much more bullish than the price (pricing_gap > threshold)
    if prob_up > 0.51 and ev_up > 0.02:
        if pricing_gap > 0.03:
            side = 'Up'
            # Scale bet by the magnitude of the pricing gap
            # A gap of 0.05 (5%) results in a 0.5 bet; 0.1 results in 1.0.
            bet_fraction = min(1.0, pricing_gap * 10.0)

    # --- DOWN SIDE LOGIC ---
    # We want: Model is bearish (prob_down > 0.5) AND 
    # there is positive EV AND 
    # the order book is much more bearish than the price (pricing_gap < -threshold)
    # Note: if micro_prob is 0.1 and cost_up is 0.5, gap is -0.4.
    elif prob_down > 0.51 and ev_down > 0.02:
        if pricing_gap < -0.03:
            side = 'Down'
            # Scale bet by the magnitude of the negative pricing gap
            bet_fraction = min(1.0, abs(pricing_gap) * 10.0)

    # Final safety clamp and minimum bet threshold
    if side and bet_fraction > 0.1:
        return side, bet_fraction

    return None, 0.0