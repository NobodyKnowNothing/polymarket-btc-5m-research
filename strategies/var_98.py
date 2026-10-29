# Var 98 | BT: -0.564 | Live PnL: $-1.71 | Trades: 75
# Desc: Implements "Coherence-Gated Momentum & Absorption (CGMA)," which classifies trades into Momentum, Absorption, or Neutral regimes by analyzing the alignment between volatility-normalized delta velocity
# Stats: {"live_returns": [-1.0, 3.303756, 1.28551, 1.890848, -1.0, 0.07774192094802856, -1.0, 0.028446435928344727, 1.1375739999999999, 1.5191249999999998, -1.0, 1.4562249999999999, 1.0471906661987305, 1.3396370000000002, 0.15118699999999996, 0.10316799999999993, 2.280573, -1.0, -1.0, 1.4562249999999999, 0.11777591705322266, 1.233902, 0.058005690574645996, -1.0, 0.07603859901428223, 1.6554160000000002, -1.0, -1.0, -1.0, 0.13876599999999994, -1.0, 2.795066, 0.06987699999999997, -1.0, -1.0, 0.05248481035232544, -1.0, -1.0, -0.8054814338684082, 0.07538449764251709, 0.09182599999999996, -1.0, -0.8870551586151123, -1.0, 0.13876599999999994, -1.0, 4.550646, 3.1194439999999997, -1.0, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.058005690574645996, 1.28551, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -0.8551623821258545, -1.0, -1.0, -1.0, -0.6810232996940613, -1.0, -1.0], "live_pnl": -1.71200031042099, "live_wins": 34, "live_total": 75, "age": 136, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.5639434998363984, "backtest_acc": 0.5135072193758733, "backtest_n": 3418}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the velocity of order flow and price movement relative to volatility.
    """
    df = df.copy()

    # 1. Volatility-Normalized Velocity Features
    # We look at the change (velocity) in delta and price over the last snapshot.
    # Normalizing by volatility makes these signals comparable across different market regimes.
    vol = df['mean_volatility'] + 1e-6
    
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['price_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    
    df['norm_delta_vel'] = df['delta_vel'] / vol
    df['norm_price_vel'] = df['price_vel'] / vol

    # 2. Delta Acceleration
    # Captures the "impulse" or the rate at which aggressive flow is increasing.
    df['delta_accel'] = df.groupby('market_slug')['delta_vel'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Coherence-Gated Momentum & Absorption (CGMA)".

    Logic:
    - Momentum Regime: Delta velocity and Price velocity are aligned (Coherence > 0) 
      AND the model's direction matches the flow. (Low EV threshold).
    - Absorption Regime: Delta velocity and Price velocity are fighting (Coherence < 0)
      AND the delta is significant. (High EV threshold for reversal/exhaustion).
    - Neutral Regime: Low coherence or low velocity. (Moderate EV threshold).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        side_sign = 1
    else:
        best_ev = ev_down
        side = 'Down'
        side_sign = -1

    # Extract engineered features
    n_delta_vel = features_row.get('norm_delta_vel', 0.0)
    n_price_vel = features_row.get('norm_price_vel', 0.0)
    
    # Coherence: +1 if flow and price move together, -1 if they fight, 0 if one is stagnant
    coherence = np.sign(n_delta_vel) * np.sign(n_price_vel)
    
    # 1. Identify Regime
    # Momentum: Flow and Price are moving in the same direction
    is_momentum = (coherence > 0) and (np.sign(n_delta_vel) == side_sign)
    
    # Absorption: Flow is strong but price is moving against it
    is_absorption = (coherence < 0) and (abs(n_delta_vel) > 0.5)

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_momentum:
        # High conviction in trend; take smaller edges
        req_ev = 0.020
        multiplier = 1.0
    elif is_absorption:
        # High risk of being caught in a squeeze; require higher edge to bet on the reversal
        req_ev = 0.045
        multiplier = 0.5
    else:
        # Noise or vacuum regime
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating
    # As the window closes, volatility spikes and signals decay faster.
    if seconds_remaining < 60:
        req_ev *= 1.3
        multiplier *= 0.8

    # 4. Final Decision
    if best_ev > req_ev:
        # Scale bet size by how much the edge exceeds the requirement
        # We normalize scaling against a 4% edge target
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0