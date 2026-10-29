# Var 187 | BT: 2.698 | Live PnL: $0.00 | Trades: 0
# Desc: Implements "Friction-Induced Reversal (FIR)," which targets high-conviction mean-reversion opportunities by identifying "absorption" events—where aggressive market orders (Delta) are being absorbed by
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 59, "rounds_underwater": 0, "abstain_rounds": 59, "backtest_score": 2.6979357260080263, "backtest_acc": 0.5628248790105753, "backtest_n": 39}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically
    calculating the efficiency of order flow and the mismatch between
    aggressive delta and order book intent.
    """
    df = df.copy()

    # 1. Price Velocity (Percentage change within the market window)
    # We use groupby to ensure we don't calculate diffs across different market windows.
    df['price_change_pct'] = df.groupby('market_slug')['binance_price'].pct_change().fillna(0)

    # 2. Flow Efficiency: How much does 1 unit of Delta move the price?
    # Low efficiency implies high absorption (the "wall" effect).
    df['flow_efficiency'] = df['price_change_pct'].abs() / (df['mean_delta'].abs() + 1e-7)

    # 3. Intent-Action Mismatch: Do Delta and OBI point in opposite directions?
    # A negative value indicates aggressive flow is hitting a thick limit order wall.
    df['obi_delta_mismatch'] = df['mean_delta'] * df['mean_obi']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting on microstructure noise."""
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
    Decide whether to trade using "Friction-Induced Reversal (FIR)".

    Logic:
    1. Detect Absorption:
       - High absolute Delta (significant aggressive flow).
       - Low price_change_pct (price is not moving).
       - Negative obi_delta_mismatch (flow is hitting a limit order wall).
    2. Direction:
       - If Delta is negative and OBI is positive (Sellers hitting Bids) -> Reversal is 'Up'.
       - If Delta is positive and OBI is negative (Buyers hitting Asks) -> Reversal is 'Down'.
    3. Execution:
       - Only trade if the model's predicted probability aligns with the reversal direction.
       - Scale bet size by model confidence and time remaining (decaying as the window closes).
    """
    # Exit if too little time remains
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    price_change = features_row.get('price_change_pct', 0.0)
    mismatch = features_row.get('obi_delta_mismatch', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)

    # --- ABSORPTION DETECTION ---
    # Thresholds: Significant flow, little movement, and clear direction mismatch
    is_absorbing = (
        abs(mean_delta) > 0.2 and 
        abs(price_change) < 0.0005 and 
        mismatch < -0.02
    )

    if not is_absorbing:
        return None, 0.0

    side = None
    
    # --- REVERSAL LOGIC ---
    # If sellers are being absorbed by bids, we bet Up
    if mean_delta < 0 and mean_obi > 0:
        if prob_up > 0.52 and ev_up > 0.02:
            side = 'Up'
            
    # If buyers are being absorbed by asks, we bet Down
    elif mean_delta > 0 and mean_obi < 0:
        if prob_down > 0.52 and ev_down > 0.02:
            side = 'Down'

    if side:
        # Confidence-based sizing: scale by (P - 0.5)
        # Time-based sizing: scale by (seconds_remaining / 300)
        confidence = (prob_up if side == 'Up' else prob_down) - 0.5
        time_factor = min(1.0, seconds_remaining / 300.0)
        
        # Base bet is scaled by confidence (0.05 confidence -> 0.1 base bet)
        bet_fraction = (confidence * 2.0) * time_factor
        
        # Clamp bet fraction to reasonable bounds
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0