# Var 102 | BT: -0.385 | Live PnL: $-3.25 | Trades: 61
# Desc: Implements "Microstructure-Sentiment Divergence Gating (MSDG)," which identifies alpha by detecting when aggressive exchange-level order flow (Delta $\times$ OBI) is decoupled from the Polymarket sent
# Stats: {"live_returns": [3.1194439999999997, 0.06254702806472778, 0.13876599999999994, -0.9766287803649902, 2.795066, -0.9084683656692505, 1.2955102920532227, -0.4744209349155426, -1.0, 0.1769179999999999, -1.0, -1.0, -1.0, 0.11476700000000006, 0.06034660339355469, -1.0, 0.04546111822128296, 2.39541, 0.16389999999999993, -1.0, -0.34636980295181274, -1.0, 0.09816926717758179, -1.0, 3.1194439999999997, -0.9129946231842041, 0.07692354917526245, -1.0, -1.0, -0.8253816366195679, -0.5249437093734741, 2.39541, 0.045158207416534424, -0.5491282939910889, -1.0, 1.4562249999999999, -0.9069671034812927, -1.0, -1.0, -1.0, -0.3870755434036255, -1.0, -1.0, 1.9791409999999998, -1.0, -0.6697772145271301, 0.11149799823760986, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, 0.05977952480316162, -0.6583038568496704], "live_pnl": -3.2489981651306152, "live_wins": 24, "live_total": 61, "age": 100, "rounds_underwater": 24, "abstain_rounds": 0, "backtest_score": -0.3850193792426959, "backtest_acc": 0.5724654116579724, "backtest_n": 3389}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically 
    calculating the divergence between exchange-level microstructure 
    and Polymarket sentiment.
    """
    df = df.copy()

    # 1. Microstructure Signal: Combined strength of aggressive flow and book imbalance
    # This represents the "real" immediate pressure on the order book.
    df['micro_signal'] = df['mean_delta'] * df['mean_obi']

    # 2. Sentiment Signal: Normalized Polymarket cost (scaled to match micro_signal magnitude)
    # We scale by 10.0 to bring the [-0.5, 0.5] range closer to typical delta/obi products.
    df['sentiment_signal'] = (df['pm_cost_up'] - 0.5) * 10.0

    # 3. Divergence: The gap between what the exchanges are doing and what the PM price reflects.
    # High positive = Aggressive buying on exchanges not yet reflected in PM price.
    # High negative = Aggressive selling on exchanges not yet reflected in PM price.
    df['divergence'] = df['micro_signal'] - df['sentiment_signal']

    # 4. Standard momentum and volatility features
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['price_imbalance'] = df['obi_spread'] / (df['price_std'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on signal over noise."""
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
    Decide whether to trade using "Microstructure-Sentiment Divergence Gating (MSDG)".

    Logic:
    - The model provides the baseline probability.
    - We use the 'divergence' feature to decide if we should lower our entry 
      threshold (trusting the momentum) or raise it (expecting a reversal/noise).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    divergence = features_row.get('divergence', 0.0)
    
    # Define thresholds based on divergence regime
    # If divergence is high, the 'true' direction is likely aligned with the micro-signal.
    if divergence > 5.0:
        # Bullish Microstructure Lag: We favor 'Up' and lower the required EV
        req_ev_up = 0.020
        req_ev_down = 0.060 # High bar for counter-trend
        side_preference = 'Up'
    elif divergence < -5.0:
        # Bearish Microstructure Lag: We favor 'Down' and lower the required EV
        req_ev_up = 0.060 # High bar for counter-trend
        req_ev_down = 0.020
        side_preference = 'Down'
    else:
        # Neutral/Converged: Require high conviction from the model alone
        req_ev_up = 0.045
        req_ev_down = 0.045
        side_preference = None

    # Decision Logic
    decision_side = None
    best_ev = 0.0

    if side_preference == 'Up':
        if ev_up > req_ev_up:
            decision_side = 'Up'
            best_ev = ev_up
        elif ev_down > req_ev_down:
            decision_side = 'Down'
            best_ev = ev_down
    elif side_preference == 'Down':
        if ev_down > req_ev_down:
            decision_side = 'Down'
            best_ev = ev_down
        elif ev_up > req_ev_up:
            decision_side = 'Up'
            best_ev = ev_up
    else:
        if ev_up > req_ev_up:
            decision_side = 'Up'
            best_ev = ev_up
        elif ev_down > req_ev_down:
            decision_side = 'Down'
            best_ev = ev_down

    if decision_side is None:
        return None, 0.0

    # Bet Sizing
    # 1. Scale by edge magnitude (relative to 5% baseline)
    # 2. Scale by divergence magnitude (higher divergence = higher conviction)
    # 3. Time decay (reduce exposure as the window closes)
    time_decay = min(1.0, seconds_remaining / 300.0)
    div_multiplier = 1.0 + min(1.0, abs(divergence) / 10.0)
    
    bet_fraction = (best_ev / 0.05) * div_multiplier * time_decay
    
    return decision_side, min(1.0, max(0.0, bet_fraction))