# Var 55 | BT: -1.172 | Live PnL: $-24.15 | Trades: 61
# Desc: ROI-Centric Squeeze-Aware Gating (RCSAG): Prioritizes trades with the highest Return on Investment (ROI) rather than raw Expected Value (EV), and applies an asymmetric hurdle to avoid betting into cro
# Stats: {"live_returns": [4.550646, -1.0, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.09417843818664551, -1.0, 2.39541, 0.03823253512382507, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, 1.1230264902114868, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -24.150249481201172, "live_wins": 12, "live_total": 61, "age": 97, "rounds_underwater": 54, "abstain_rounds": 0, "backtest_score": -1.171868839008909, "backtest_acc": 0.584002794271743, "backtest_n": 2233}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, emphasizing 
    confluence between flow, book pressure, and funding.
    """
    df = df.copy()

    # 1. Flow-Book Confluence: Is the aggressive flow aligned with the book imbalance?
    df['obi_delta_confluence'] = df['mean_obi'] * df['sum_delta']
    
    # 2. Funding-Flow Confluence: Is the aggressive flow paying or receiving funding?
    # (High positive: aggressive buyers are paying; High negative: aggressive sellers are paying)
    df['funding_delta_confluence'] = df['mean_funding'] * df['sum_delta']

    # 3. Volatility-Adjusted Flow: Normalized aggressive flow per unit of volatility
    df['vol_adjusted_delta'] = df['sum_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Delta Velocity: The acceleration of aggressive trade flow within the window
    df['delta_velocity'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)

    # 5. Price Efficiency: How much price moves per unit of delta
    df['price_delta_efficiency'] = df['sum_delta'] / (df['price_std'] + 1e-6)

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    # We keep pm_cost_up/down as they help the model learn the 'mispricing' relative to exchange data
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == float:
            # Cap extreme outliers to prevent XGBoost instability
            X.loc[X[col] > 1e10, col] = 0
            X.loc[X[col] < -1e10, col] = 0

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "ROI-Centric Squeeze-Aware Gating (RCSAG)".
    Instead of raw EV, we optimize for ROI (Return on Investment) and 
    apply higher hurdles when funding suggests a crowded squeeze is imminent.
    """
    # 1. Calculate ROI (Expected Return / Cost)
    # We use a small epsilon to prevent division by zero
    eps = 1e-4
    roi_up = (prob_up - cost_up) / max(cost_up, eps)
    roi_down = (prob_down - cost_down) / max(cost_down, eps)

    if roi_up > roi_down:
        side = 'Up'
        best_roi = roi_up
    else:
        side = 'Down'
        best_roi = roi_down

    # 2. Base Hurdle (Targeting ~8% ROI)
    hurdle = 0.08

    # 3. Squeeze-Aware Hurdle Adjustment
    # If funding is extremely positive, 'Up' trades are risky (potential long squeeze)
    # If funding is extremely negative, 'Down' trades are risky (potential short squeeze)
    funding = features_row.get('mean_funding', 0.0)
    
    if side == 'Up' and funding > 0.0002:
        hurdle += 0.06  # Increase hurdle for longs in a long-heavy market
    elif side == 'Down' and funding < -0.0002:
        hurdle += 0.06  # Increase hurdle for shorts in a short-heavy market

    # 4. Execution Decision
    if best_roi < hurdle:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale bet size by:
    # - Excess ROI (how much above hurdle)
    # - Time decay (reduce size as we approach the 5-min resolution to avoid last-second noise)
    time_factor = max(0.1, seconds_remaining / 300.0)
    roi_multiplier = best_roi / hurdle
    
    bet_fraction = roi_multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to avoid tiny, inefficient trades
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction