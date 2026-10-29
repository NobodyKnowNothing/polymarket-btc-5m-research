# Var 149 | BT: -0.466 | Live PnL: $-17.34 | Trades: 60
# Desc: Targets "Coiled Spring" breakouts by identifying high-absorption regimes (high aggressive delta relative to price dispersion) that subsequently align with price velocity.
# Stats: {"live_returns": [1.043916940689087, -1.0, -1.0, -1.0, -1.0, 2.950496, -1.0, -1.0, -1.0, -1.0, 1.043916940689087, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, 1.2535656690597534, 2.073133, -1.0, 2.651597, -0.9176695346832275, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9547239542007446, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, 2.651597, -1.0, 0.04551428556442261, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, 0.12663000000000002, 2.651597], "live_pnl": -17.33507537841797, "live_wins": 16, "live_total": 60, "age": 107, "rounds_underwater": 58, "abstain_rounds": 0, "backtest_score": -0.466452062436373, "backtest_acc": 0.5878717651602936, "backtest_n": 3713}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'Absorption-Breakout' cycle: high delta with low price movement 
    followed by velocity alignment.
    """
    df = df.copy()

    # 1. Absorption Index: High delta with low cross-exchange dispersion 
    # suggests aggressive orders are being absorbed by limit orders.
    # We use price_std as a proxy for the 'resistance' level.
    df['absorption_index'] = df['mean_delta'].abs() / (df['price_std'] + 1e-6)

    # 2. Price Velocity: The speed at which the price is moving away from the strike.
    # Normalized by time remaining to capture momentum intensity.
    df['price_velocity'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1e-6)

    # 3. Delta-Price Efficiency: How much price moves per unit of delta.
    # Low efficiency = Absorption; High efficiency = Liquidity Vacuum/Breakout.
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 4. OBI-Delta Congruence: Do the book intent and actual flow agree?
    df['obi_delta_sync'] = df['mean_obi'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Absorption-Breakout (MAB)".

    Logic:
    1. Identify "Coiled Spring" (Absorption): 
       High absorption_index (aggressive flow vs low price dispersion) 
       AND the price velocity is starting to align with the delta direction.
    2. Identify "Standard Momentum":
       If not in an absorption breakout, use a higher EV threshold to avoid noise.
    3. Dynamic Sizing:
       Scale bet size based on the magnitude of the edge and the strength of the absorption.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    absorption = features_row.get('absorption_index', 0.0)
    velocity = features_row.get('price_velocity', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    
    delta_dir = 1.0 if mean_delta > 0 else (-1.0 if mean_delta < 0 else 0.0)
    velocity_dir = 1.0 if velocity > 0 else (-1.0 if velocity < 0 else 0.0)

    side = None
    best_ev = 0.0
    bet_fraction = 0.0

    # --- REGIME 1: THE COILED SPRING (Breakout from Absorption) ---
    # High absorption + direction alignment = high conviction breakout
    is_coiled_spring = (absorption > 1.5) and (delta_dir == velocity_dir) and (delta_dir != 0)

    if is_coiled_spring:
        if delta_dir > 0 and ev_up > 0.02:
            side = 'Up'
            best_ev = ev_up
        elif delta_dir < 0 and ev_down > 0.02:
            side = 'Down'
            best_ev = ev_down
        
        if side:
            # Scale by EV and the 'tightness' of the spring (absorption)
            # We use log1p to prevent extreme absorption values from exploding the bet
            scaling = (best_ev / 0.03) * (math.log1p(absorption) / 2.0)
            bet_fraction = min(1.0, max(0.1, scaling))

    # --- REGIME 2: STANDARD FLOW (Trend Following) ---
    else:
        # Higher hurdle for non-absorption trades to protect Sharpe
        if ev_up > 0.04:
            side = 'Up'
            best_ev = ev_up
            bet_fraction = min(1.0, best_ev / 0.05)
        elif ev_down > 0.04:
            side = 'Down'
            best_ev = ev_down
            bet_fraction = min(1.0, best_ev / 0.05)

    # Final safety: reduce bet as time runs out to avoid last-second volatility
    if side and seconds_remaining < 30:
        bet_fraction *= (seconds_remaining / 30.0)

    return (side, bet_fraction) if side else (None, 0.0)