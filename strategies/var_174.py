# Var 174 | BT: -0.138 | Live PnL: $-2.19 | Trades: 7
# Desc: Implements "Microstructure Triple-Congruence (MTC)," which requires simultaneous alignment of Price, Delta, and OBI to validate model-predicted EV, effectively filtering out "fake" moves caused by low
# Stats: {"live_returns": [-0.06605994701385498, -0.05, -1.0, -0.3535158634185791, 0.002967972308397293, -0.08623659610748291, -0.6336266994476318], "live_pnl": -2.1864709854125977, "live_wins": 1, "live_total": 7, "age": 152, "rounds_underwater": 7, "abstain_rounds": 59, "backtest_score": -0.13765595585224902, "backtest_acc": 0.5531095211887727, "backtest_n": 270}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronization of price movement, aggressive flow, and book intent.
    """
    df = df.copy()

    # 1. Microstructure Momentum: The product of OBI and Delta.
    # High positive = Strong trend; High negative = Strong absorption/reversal.
    df['microstructure_momentum'] = df['mean_obi'] * df['mean_delta']

    # 2. Price-Flow Synchronization (Temporal Diffs)
    # We use groupby to ensure diffs are only calculated within the same market window.
    group = df.groupby('market_slug')
    df['price_change'] = group['binance_price'].diff().fillna(0)
    df['delta_change'] = group['mean_delta'].diff().fillna(0)
    df['obi_change'] = group['mean_obi'].diff().fillna(0)

    # 3. Price Impact Coefficient: How much price moves per unit of Delta.
    # High values indicate "Vacuum" regimes (low liquidity, high slippage).
    # Low values indicate "Efficient" regimes (high liquidity, sustainable moves).
    df['price_impact'] = df['price_change'] / (df['mean_delta'].abs() + 1e-6)

    # 4. OBI-Delta Acceleration: Are intent and flow ramping up together?
    df['intent_flow_accel'] = df['obi_change'] * df['delta_change']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return robust hyperparameters with high regularization to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Triple-Congruence (MTC)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Triple-Congruence Gate: A trade is only valid if:
       - Model predicts an edge (EV > 0.03).
       - Price, Delta, and OBI all agree on the direction (Congruence).
       - The move is not a "Vacuum" (Price Impact is within reasonable bounds).
    3. Dynamic Sizing: Scale bet by EV and inversely by Price Impact.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    price_change = features_row.get('price_change', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    price_impact = features_row.get('price_impact', 0.0)

    side = None
    best_ev = 0.0
    
    # Minimum EV threshold to consider a trade
    min_ev_threshold = 0.03

    # 1. Triple-Congruence Logic
    # We look for 'Up' trades where Price, Delta, and OBI are all positive.
    if ev_up > min_ev_threshold:
        if price_change > 0 and mean_delta > 0 and mean_obi > 0:
            side = 'Up'
            best_ev = ev_up
            
    # We look for 'Down' trades where Price, Delta, and OBI are all negative.
    elif ev_down > min_ev_threshold:
        if price_change < 0 and mean_delta < 0 and mean_obi < 0:
            side = 'Down'
            best_ev = ev_down

    # 2. Execution and Scaling
    if side:
        # Scaling factors:
        # a) EV Magnitude: Target 4% EV for full bet.
        # b) Liquidity Quality: Scale down if price_impact is high (avoiding vacuum/slippage).
        # c) Time Decay: Be more cautious as the window closes.
        
        ev_scale = best_ev / 0.04
        liquidity_scale = 1.0 / (1.0 + abs(price_impact))
        time_scale = min(1.0, seconds_remaining / 120.0) if seconds_remaining < 120 else 1.0
        
        bet_fraction = ev_scale * liquidity_scale * time_scale
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0