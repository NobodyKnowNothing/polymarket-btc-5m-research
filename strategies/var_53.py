# Var 53 | BT: 0.392 | Live PnL: $-22.23 | Trades: 61
# Desc: Implements "Regime-Switching Signal Selection (RSSS)," which dynamically switches the primary alpha source between aggressive flow (Delta) in high-volatility regimes and passive positioning (OBI) in l
# Stats: {"live_returns": [-1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.6554160000000002, 4.550646, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.16389999999999993, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, 1.0081479999999998, -1.0, 1.1846420000000002, -1.0, 0.038675000000000015, -1.0, -1.0, -1.0, 0.05925399999999992, 1.28551, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, 0.11476700000000006, -1.0, -1.0, 0.15118699999999996, 0.15118699999999996, -1.0, 0.09182599999999996, -1.0, -0.7861500382423401, -1.0, 0.05769532918930054, 0.09182599999999996, -1.0, 1.1375739999999999, -1.0], "live_pnl": -22.233980178833008, "live_wins": 19, "live_total": 61, "age": 135, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": 0.3916210670646643, "backtest_acc": 0.560427807486631, "backtest_n": 1557}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    volatility-normalized momentum and volatility acceleration.
    """
    df = df.copy()

    # 1. Volatility Acceleration: Detect if we are entering a high-vol regime
    df['vol_acceleration'] = df.groupby('market_slug')['mean_volatility'].diff().fillna(0)

    # 2. Volatility-Normalized Flow (Delta): How much aggressive buying per unit of vol
    df['delta_vol_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Volatility-Normalized Passive Pressure (OBI): How much book imbalance per unit of vol
    df['obi_vol_ratio'] = df['mean_obi'] / (df['mean_volatility'] + 1e-6)

    # 4. Price-Delta Efficiency: How much price moves relative to aggressive flow
    # (High efficiency = thin book/breakout; Low efficiency = absorption)
    df['price_delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == float:
            X.loc[X[col] > 1e10, col] = 0
            X.loc[X[col] < -1e10, col] = 0

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 40.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Switching Signal Selection (RSSS)".
    In high volatility, we look for aggressive momentum (Delta).
    In low volatility, we look for passive positioning (OBI).
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Identify Regime
    # We use mean_volatility as the regime switch. 
    # Note: Threshold is tuned for BTC 5-min volatility scales.
    vol = features_row.get('mean_volatility', 0.0)
    is_high_vol = vol > 0.0015 

    # 3. Regime-Specific Signal Gating
    # High Vol: We need aggressive Delta to confirm a breakout.
    # Low Vol: We need OBI to confirm a steady drift.
    if is_high_vol:
        signal_strength = abs(features_row.get('mean_delta', 0.0))
        signal_threshold = 0.0025
        hurdle = 0.035  # Higher hurdle in high vol to avoid whipsaws
    else:
        signal_strength = abs(features_row.get('mean_obi', 0.0))
        signal_threshold = 0.25
        hurdle = 0.018  # Lower hurdle in low vol to capture drifts

    # 4. Execution Logic
    if best_ev < hurdle or signal_strength < signal_threshold:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale by edge, signal strength, and time remaining
    time_factor = max(0.2, seconds_remaining / 300.0)
    
    # Normalized signal (how much above threshold we are)
    signal_multiplier = signal_strength / signal_threshold
    
    # Normalized edge (how much above hurdle we are)
    edge_multiplier = best_ev / hurdle

    bet_fraction = edge_multiplier * signal_multiplier * time_factor
    
    # Cap bet fraction at 1.0
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction