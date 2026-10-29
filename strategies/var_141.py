# Var 141 | BT: -1.265 | Live PnL: $-33.39 | Trades: 61
# Desc: Implements "Convergent Momentum Gating (CMG)," which differentiates between "High-Conviction Breakouts" (where price moves in the predicted direction while cross-exchange spreads narrow) and "Fragile
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7904570698738098, 1.233902, -1.0, -1.0, 0.1500207781791687, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, 0.02803635597229004, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -33.39006423950195, "live_wins": 12, "live_total": 61, "age": 110, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -1.2647734520225218, "backtest_acc": 0.548368069209595, "backtest_n": 3839}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the velocity of price movement relative to exchange spread convergence.
    """
    X = df.copy()

    # 1. Price and Spread Velocity (Grouped by market to prevent leakage)
    # We look at the rate of change of the price and the rate of change of the spread.
    X['price_vel'] = X.groupby('market_slug')['binance_price'].diff().fillna(0)
    X['spread_vel'] = X.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 2. Convergence Index: High if price is moving and spreads are narrowing.
    # A narrowing spread (negative spread_vel) during a price move (positive price_vel)
    # indicates the move is being validated across all venues.
    X['convergence_index'] = X['price_vel'] * (-X['spread_vel'])

    # 3. Flow Acceleration: Is the aggressive flow and OBI increasing or decreasing?
    X['delta_accel'] = X.groupby('market_slug')['mean_delta'].diff().fillna(0)
    X['obi_accel'] = X.groupby('market_slug')['mean_obi'].diff().fillna(0)
    X['flow_accel'] = X['delta_accel'] + X['obi_accel']

    # 4. Momentum-Efficiency: How much price moves per unit of delta.
    # High efficiency = price moves easily (low friction/absorption).
    X['delta_efficiency'] = X['price_vel'] / (X['mean_delta'].abs() + 1e-6)
    X['obi_efficiency'] = X['price_vel'] / (X['mean_obi'].abs() + 1e-6)
    X['microstructure_efficiency'] = (X['delta_efficiency'] + X['obi_efficiency']) / 2.0

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

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
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Convergent Momentum Gating (CMG)".

    Logic:
    - Identify the side with the highest Expected Value (EV).
    - Determine if the move is "Aligned" (Price moving in direction + spreads narrowing + flow accelerating).
    - Determine if the move is "Divergent" (Price moving in direction but spreads widening).
    - Apply asymmetric thresholds:
        - Aligned: Low EV threshold (0.025), High bet size.
        - Divergent: High EV threshold (0.055), Low bet size.
        - Neutral: Standard EV threshold (0.04), Standard bet size.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    conv = features_row.get('convergence_index', 0.0)
    flow_acc = features_row.get('flow_accel', 0.0)
    
    # 1. Alignment Check
    # For 'Up': we want price_vel > 0 (conv > 0) and flow_acc > 0.
    # For 'Down': we want price_vel < 0 (conv > 0) and flow_acc < 0.
    # Note: convergence_index = price_vel * -spread_vel. 
    # If price_vel is negative and spread_vel is negative, conv is positive.
    # So: side == 'Up' and conv > 0 and flow_acc > 0
    #     side == 'Down' and conv > 0 and flow_acc < 0
    
    is_aligned = (side == 'Up' and conv > 0 and flow_acc > 0) or \
                 (side == 'Down' and conv > 0 and flow_acc < 0)
    
    # 2. Divergence Check
    # If convergence is negative, it means price and spread are moving in the same direction
    # (e.g., price up, spread up), which suggests instability/fragility.
    is_divergent = (side == 'Up' and conv < 0) or \
                   (side == 'Down' and conv < 0)

    # 3. Asymmetric Thresholding
    if is_aligned:
        threshold = 0.025
        bet_multiplier = 1.4
    elif is_divergent:
        threshold = 0.055
        bet_multiplier = 0.5
    else:
        threshold = 0.04
        bet_multiplier = 1.0

    # 4. Execution
    if best_ev > threshold:
        # Scale bet by edge and multiplier, capped at 1.0
        # We target a 4% edge as the baseline for a 1.0 bet
        bet_fraction = (best_ev / 0.04) * bet_multiplier
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0