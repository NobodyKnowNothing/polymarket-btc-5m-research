# Var 82 | BT: -0.133 | Live PnL: $-1.75 | Trades: 76
# Desc: Implements "Microstructure Convergence & Flow Acceleration (MCFA)," which filters for high-conviction trades where exchange price dispersion is contracting (convergence) and aggressive delta is increa
# Stats: {"live_returns": [-1.0, -1.0, 2.073133, -1.0, -1.0, 3.303756, -1.0, 1.5854279999999998, 1.3964720000000002, 0.15118699999999996, 0.08073200000000003, 0.06389427185058594, -1.0, 1.233902, -1.0, -1.0, 1.1375739999999999, 0.12663000000000002, -1.0, -1.0, -0.903390109539032, 3.505625, 2.9236152172088623, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, 1.2413816452026367, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, 6.840158462524414, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9126347899436951, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.7482709884643555, "live_wins": 24, "live_total": 76, "age": 158, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.13335821670344805, "backtest_acc": 0.5714643931795386, "backtest_n": 2487}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    second-order microstructure dynamics (velocity of spread and flow).
    """
    df = df.copy()

    # 1. Price Spread Velocity: Is the dispersion between exchanges narrowing?
    # Negative = convergence (market settling on a price), Positive = divergence (uncertainty).
    df['price_spread_velocity'] = df.groupby('market_slug')['price_spread'].diff()

    # 2. Delta Acceleration: Is the net aggressive flow increasing in magnitude?
    # Positive = buying pressure is ramping up, Negative = selling pressure is ramping up.
    df['sum_delta_velocity'] = df.groupby('market_slug')['sum_delta'].diff()

    # 3. OBI Velocity: Is the order book imbalance shifting rapidly?
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff()

    # 4. Price Velocity: Rate of change in the lead exchange (Binance).
    df['binance_price_velocity'] = df.groupby('market_slug')['binance_price'].diff()

    # 5. Momentum Acceleration: Is the momentum trend strengthening?
    df['momentum_velocity'] = df.groupby('market_slug')['mean_momentum'].diff()

    # Drop metadata and target
    # We keep seconds_remaining as it can be a useful feature for the model to learn time-decay.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle NaNs from diff() and prevent inf
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on stable microstructure signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 35.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Convergence & Flow Acceleration (MCFA)".

    Logic:
    - Identifies "Structural Moves" vs "Noisy Moves".
    - A structural move is characterized by:
        1. Convergence: The price spread between exchanges is shrinking (price discovery).
        2. Acceleration: The aggressive delta is increasing in the direction of the trade.
    - Effective EV is scaled heavily by these two factors.
    - If the market is diverging (spread widening) or delta is stalling, we abstain 
      even if the model predicts a high probability.
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
    spread_vel = features_row.get('price_spread_velocity', 0.0)
    delta_vel = features_row.get('sum_delta_velocity', 0.0)

    # 1. Convergence Factor: High if spread is narrowing (price discovery in progress)
    # We use a multiplier approach to penalize divergence.
    convergence_mult = 1.4 if spread_vel < 0 else 0.5

    # 2. Acceleration Factor: High if delta velocity aligns with the trade direction
    # If we are going 'Up', we want delta_vel > 0.
    flow_mult = 1.0
    if side == 'Up' and delta_vel > 0:
        flow_mult = 1.4
    elif side == 'Down' and delta_vel < 0:
        flow_mult = 1.4
    else:
        flow_mult = 0.5

    # 3. Calculate Effective EV
    effective_ev = best_ev * convergence_mult * flow_mult

    # 4. Dynamic Thresholding
    # We require a higher threshold as we approach the end of the window to avoid volatility spikes.
    base_threshold = 0.035
    if seconds_remaining < 60:
        base_threshold = 0.055
    elif seconds_remaining < 150:
        base_threshold = 0.045

    if effective_ev > base_threshold:
        # Scale bet fraction: target a full bet at 5% effective edge.
        # We use a sigmoid-like scaling to ensure we don't over-leverage on thin edges.
        bet_fraction = min(1.0, effective_ev / 0.05)
        return side, max(0.1, bet_fraction)

    return None, 0.0