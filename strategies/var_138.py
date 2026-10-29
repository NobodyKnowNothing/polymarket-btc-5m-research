# Var 138 | BT: 0.633 | Live PnL: $-1.54 | Trades: 6
# Desc: Implements "Microstructural Alpha-Gap Gating (MAGG)," which identifies mispriced Polymarket tokens by calculating the divergence between microstructural pressure (a tanh-squashed blend of OBI and Delt
# Stats: {"live_returns": [0.10316799999999993, -1.0, 0.10316799999999993, -1.0, 0.10316799999999993, 0.15118699999999996], "live_pnl": -1.5393090000000003, "live_wins": 4, "live_total": 6, "age": 174, "rounds_underwater": 5, "abstain_rounds": 61, "backtest_score": 0.633257177653073, "backtest_acc": 0.5330009970089731, "backtest_n": 315}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    divergence between microstructural pressure and Polymarket's implied price.
    """
    df = df.copy()

    # 1. Microstructural Pressure: A squashed [-1, 1] metric combining OBI and Delta.
    # We use tanh to ensure extreme values don't blow out the model while maintaining direction.
    # OBI is weighted 50%, Delta is weighted 50%.
    df['micro_pressure'] = (
        np.tanh(df['mean_obi']) * 0.5 + 
        np.tanh(df['mean_delta'] / 10.0) * 0.5
    )

    # 2. Polymarket Implied Pressure: Converts the cost of the 'Up' token into a [-1, 1] signal.
    # If cost_up is 0.5, pressure is 0. If cost_up is 0.1 (very bullish), pressure is 0.8.
    # If cost_up is 0.9 (very bearish), pressure is -0.8.
    df['pm_pressure'] = (1.0 - df['pm_cost_up']) * 2.0 - 1.0

    # 3. The Alpha Gap: The core signal. 
    # Positive = Microstructure is much more bullish than Polymarket price.
    # Negative = Microstructure is much more bearish than Polymarket price.
    df['pressure_gap'] = df['micro_pressure'] - df['pm_pressure']

    # 4. Microstructural Acceleration: Change in delta momentum within the window.
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_accel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on the Alpha Gap signal."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructural Alpha-Gap Gating (MAGG)".

    Logic:
    - Calculate EV for both sides.
    - Identify the 'Alpha Gap': The difference between exchange-driven pressure 
      and Polymarket's implied probability.
    - Thresholding: We only trade if the model sees EV AND the Alpha Gap is wide.
    - Time-Decay: As seconds_remaining decreases, we require a much wider gap 
      to avoid being caught in late-window noise/volatility.
    - Sizing: Bet fraction scales linearly with the magnitude of the Alpha Gap.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    gap = features_row.get('pressure_gap', 0.0)

    # 1. Dynamic Thresholding based on time remaining
    # We need a significant 'mispricing' to justify the risk of a 5-min window.
    if seconds_remaining > 120:
        gap_threshold = 0.15
    elif seconds_remaining > 60:
        gap_threshold = 0.22
    else:
        gap_threshold = 0.30

    # 2. Decision Logic
    # We want EV > 0 AND a gap that confirms the direction of the EV.
    
    # Case: Bullish Trade
    if ev_up > 0.02 and gap > gap_threshold:
        # Scale bet: gap of 0.2 -> 0.3 bet; gap of 0.4 -> 0.6 bet
        bet_fraction = gap * 1.5
        return 'Up', min(1.0, max(0.1, bet_fraction))

    # Case: Bearish Trade
    if ev_down > 0.02 and gap < -gap_threshold:
        # Scale bet: gap of -0.2 -> 0.3 bet; gap of -0.4 -> 0.6 bet
        bet_fraction = abs(gap) * 1.5
        return 'Down', min(1.0, max(0.1, bet_fraction))

    return None, 0.0