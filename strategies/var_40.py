# Var 40 | BT: -0.514 | Live PnL: $-18.21 | Trades: 61
# Desc: Implements "High-Efficiency Impulse Capture (HEIC)," which filters trades by requiring alignment between aggressive flow (Delta), order book pressure (OBI), and price momentum, while scaling exposure
# Stats: {"live_returns": [1.9791409999999998, -0.8563080430030823, -0.21986715495586395, -0.6815713047981262, 0.018406540155410767, -0.697149932384491, -0.9692161679267883, 0.4321686029434204, -1.0, -1.0, -0.7259522676467896, 1.7294070000000001, -1.0, -1.0, -0.2324424833059311, -0.8074703216552734, 1.5450737476348877, -1.0, -1.0, -0.7761178612709045, -1.0, -0.32777702808380127, -1.0, -1.0, -1.0, -0.60273677110672, -1.0, -0.1235533207654953, -0.41472941637039185, 4.550646, -0.24741636216640472, 2.2127158641815186, -0.29761868715286255, -0.7365759015083313, -0.6013745069503784, -0.24687626957893372, -0.19724933803081512, -1.0, -0.4825543165206909, -0.26415035128593445, 1.3441323041915894, 0.19988314807415009, -0.3694477081298828, -1.0, 0.4130113124847412, -0.6975300312042236, -1.0, -0.36795148253440857, -1.0, -0.6099755764007568, -0.5025458335876465, -1.0, 1.7628560066223145, -0.8576226234436035, -0.14163453876972198, -0.28361910581588745, -1.0, -1.0, -1.0, -0.5618705749511719, -0.4962092339992523], "live_pnl": -18.209672927856445, "live_wins": 11, "live_total": 61, "age": 90, "rounds_underwater": 56, "abstain_rounds": 0, "backtest_score": -0.5137326889550563, "backtest_acc": 0.5135135135135135, "backtest_n": 2093}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the alignment of flow, book pressure, and momentum to identify 
    'clean' impulse moves.
    """
    # Create a copy to avoid SettingWithCopy warnings
    X = df.copy()

    # 1. Delta-OBI Sync: Measures if aggressive flow is supported by book pressure.
    # Positive = buying flow into a bid-heavy book; Negative = selling flow into an ask-heavy book.
    X['delta_obi_sync'] = X['mean_delta'] * X['mean_obi']

    # 2. Momentum-Flow Sync: Directional alignment of momentum and aggressive flow.
    X['momentum_flow_sync'] = X['mean_momentum'] * X['mean_delta']

    # 3. Flow Efficiency: How much delta is actually moving the price relative to volatility.
    # High values = efficient moves; Low values = absorption/high-volatility noise.
    X['flow_efficiency'] = X['mean_delta'].abs() / (X['mean_volatility'] + 1e-6)

    # 4. Price Stability: Inverse of cross-exchange price dispersion.
    # High values = high consensus; Low values = exchange dislocation.
    X['price_stability'] = 1.0 / (X['price_spread'] + 1e-6)

    # 5. Funding-Delta Alignment: Captures potential squeeze/reversal setups.
    X['funding_delta_sync'] = X['mean_funding'] * X['mean_delta']

    # Drop metadata and target columns to ensure no leakage
    # Note: seconds_remaining is passed separately to strategy_filter
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup for XGBoost
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters for noisy microstructure signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "High-Efficiency Impulse Capture (HEIC)".
    This strategy scales bets based on the 'cleanliness' of the move:
    requiring alignment of Delta, OBI, and Momentum, and penalizing absorption.
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
        side_sign = 1
    else:
        side = 'Down'
        best_ev = ev_down
        side_sign = -1

    # 2. Dynamic Hurdle: Increase the required edge as the window closes
    # As seconds_remaining -> 0, the hurdle increases from 0.03 to 0.05
    time_ratio = seconds_remaining / 300.0
    hurdle = 0.03 + (1.0 - time_ratio) * 0.02

    if best_ev < hurdle:
        return None, 0.0

    # 3. Extract Microstructure Metrics
    delta_obi_sync = features_row.get('delta_obi_sync', 0.0)
    momentum_flow_sync = features_row.get('momentum_flow_sync', 0.0)
    flow_efficiency = features_row.get('flow_efficiency', 0.0)
    price_stability = features_row.get('price_stability', 0.0)

    # 4. Multiplier Logic (The Alpha)
    multiplier = 1.0

    # A) Alignment Check: Is the book and momentum supporting the model's direction?
    # If we are betting 'Up', we want positive delta_obi_sync and positive momentum_flow_sync.
    if (delta_obi_sync * side_sign) <= 0:
        multiplier *= 0.5  # Penalize if flow is fighting the book
    
    if (momentum_flow_sync * side_sign) <= 0:
        multiplier *= 0.7  # Penalize if momentum is counter-directional

    # B) Efficiency Check: Is the move 'clean' or is it being absorbed?
    # If delta is high but volatility is also high, efficiency is low.
    if flow_efficiency < 0.05:
        multiplier *= 0.3  # Heavy penalty for absorption/high-vol noise

    # C) Consensus Check: Are exchanges in agreement?
    if price_stability < 50.0:
        multiplier *= 0.7  # Penalize high cross-exchange dispersion

    # 5. Execution & Sizing
    # Base bet is scaled by the edge relative to the hurdle
    # Final bet is also scaled by time remaining to reduce late-window variance
    bet_fraction = (best_ev / hurdle) * multiplier * time_ratio
    
    # Clamp bet_fraction between 0 and 1
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold for a meaningful trade
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction