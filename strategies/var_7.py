# Var 7 | BT: 1.974 | Live PnL: $0.00 | Trades: 0
# Desc: Implements "Absorption-Aware Efficiency Gating," which uses a logarithmic penalty on the absorption index (the ratio of aggressive flow to price movement) and scales the entry hurdle by volatility to
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 6, "rounds_underwater": 0, "abstain_rounds": 6, "backtest_score": 1.9740102302702265, "backtest_acc": 0.5369828666945257, "backtest_n": 3}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features by calculating 
    absorption, exchange agreement, and price efficiency.
    """
    X = df.copy()

    # 1. Convert Volatility Regime string to numeric if necessary
    if 'mean_volatility' in X.columns and X['mean_volatility'].dtype == object:
        X['mean_volatility'] = X['mean_volatility'].apply(lambda x: 0.02 if x == 'high' else 0.01)

    # 2. Absorption Index: High value means high delta but no price movement (absorption)
    # We add a small epsilon to prevent division by zero.
    X['absorption_index'] = X['sum_delta'].abs() / (X['price_vs_strike_pct'].abs() + 1e-6)

    # 3. Agreement: Inverse of the cross-exchange price spread
    X['agreement'] = 1.0 / (1.0 + X['price_spread'])

    # 4. Price Efficiency: How much delta is actually moving the price per unit of volatility
    X['efficiency'] = X['mean_delta'].abs() / (X['mean_volatility'] + 1e-6)

    # Drop metadata and target columns
    cols_to_drop = [c for c in ['market_slug', 'timestamp', 'actual_outcome'] if c in X.columns]
    X = X.drop(columns=cols_to_drop)

    # Final cleanup: handle NaNs and infinities
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for high regularization to combat noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 10.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using Absorption-Aware Efficiency Gating.
    Penalizes trades where aggressive flow is being absorbed by liquidity 
    without moving the price, and scales hurdles by volatility.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered features
    # absorption_index: high = bad (flow is being absorbed)
    # agreement: high = good (exchanges agree)
    abs_idx = features_row.get('absorption_index', 0.0)
    agree = features_row.get('agreement', 1.0)
    vol = features_row.get('mean_volatility', 0.01)
    
    # Safety check for volatility type
    if isinstance(vol, str):
        vol = 0.02 if vol == 'high' else 0.01

    # 3. Calculate Conviction Multiplier
    # We use a log-scaled penalty for absorption so it doesn't kill all trades instantly,
    # but heavily discounts them when delta is high and price movement is zero.
    absorption_penalty = 1.0 / (1.0 + math.log1p(abs_idx))
    
    # Conviction is the product of exchange consensus and the lack of absorption
    conviction = agree * absorption_penalty
    effective_ev = best_ev * conviction

    # 4. Dynamic Hurdle
    # In high volatility, we require a larger margin of safety.
    # Base hurdle 2.5% + volatility-scaled component.
    hurdle = 0.025 * (1.0 + vol * 10.0)

    # 5. Execution Logic
    if effective_ev > hurdle:
        # Time-decay: reduce exposure as the window closes to avoid last-second noise
        time_factor = max(0.2, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: scale based on how much the effective edge exceeds the hurdle.
        # We use a 0.5 coefficient to remain conservative.
        bet_fraction = min(1.0, (effective_ev / hurdle) * 0.5) * time_factor

        # Minimum bet threshold to avoid dust
        if bet_fraction < 0.02:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0