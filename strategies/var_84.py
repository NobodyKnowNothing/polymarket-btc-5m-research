# Var 84 | BT: -1.365 | Live PnL: $-2.55 | Trades: 61
# Desc: Implements "Microstructure Elasticity & Regime-Alignment Scaling (MERAS)," which uses the relationship between price velocity and order flow (elasticity) to distinguish between "Trend/Ignition" and "A
# Stats: {"live_returns": [0.025032460689544678, -1.0, 1.233902, -1.0, -1.0, -0.8363094329833984, 1.1375739999999999, -0.7781400680541992, 2.39541, -1.0, -1.0, -1.0, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 2.950496, -1.0, -1.0, -1.0, 0.025032460689544678, -0.788027822971344, 0.04885600000000001, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -0.8152734637260437, -1.0, 2.39541, -1.0, -1.0, -1.0, 0.8493208289146423, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -2.5507760047912598, "live_wins": 20, "live_total": 61, "age": 110, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": -1.3650087996281175, "backtest_acc": 0.5548081714000996, "backtest_n": 2885}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the elasticity of price relative to order flow (absorption vs ignition).
    """
    df = df.copy()

    # 1. Micro-Velocity Features (Grouped by market to prevent leakage)
    # price_change: How much the price moved in the last snapshot
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    # delta_change: How much the aggressive flow changed
    df['delta_change'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Flow Elasticity: How much price moves per unit of aggressive delta
    # Low elasticity = Absorption (High delta, low price move)
    # High elasticity = Ignition (High delta, high price move)
    df['flow_elasticity'] = df['price_change'] / (df['mean_delta'].abs() + 1e-6)

    # 3. Congruence & Volatility Scaling
    # Does the order book imbalance support the aggressive flow?
    df['obi_delta_congruence'] = df['mean_obi'] * df['mean_delta']
    # Is the flow significant relative to the current volatility?
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters tuned for detecting regime-based microstructure signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Elasticity & Regime-Alignment Scaling (MERAS)".

    Logic:
    - Identify the 'Microstructure Regime' by comparing the sign of price movement 
      to the sign of aggressive delta flow.
    - Regime 1: Trend (Ignition) -> Price and Delta move in the same direction.
    - Regime 2: Absorption (Reversal) -> Price and Delta move in opposite directions (or price is stagnant).
    - Apply a multiplier to the Expected Value (EV) based on whether the model's 
      predicted side aligns with the identified regime.
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
    mean_delta = features_row.get('mean_delta', 0.0)
    price_change = features_row.get('price_change', 0.0)
    
    flow_dir = np.sign(mean_delta)
    price_dir = np.sign(price_change)

    # Determine Regime
    # Trend: Flow and Price are aligned and moving
    is_trend = (flow_dir != 0) and (price_dir == flow_dir)
    # Absorption: Flow and Price are misaligned (including price stagnation)
    is_absorption = (flow_dir != 0) and (price_dir != flow_dir)

    # Calculate Multiplier
    # If Trend: Amplify if side matches flow, penalize if side opposes flow
    # If Absorption: Amplify if side opposes flow (reversal), penalize if side matches flow
    multiplier = 1.0
    if is_trend:
        multiplier = 1.25 if side == (('Up' if flow_dir > 0 else 'Down')) else 0.5
    elif is_absorption:
        multiplier = 1.25 if side == (('Down' if flow_dir > 0 else 'Up')) else 0.5

    effective_ev = best_ev * multiplier

    # Dynamic Thresholding
    # Tighten requirements as the market window closes to avoid noise
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 15:
        base_threshold = 0.07

    if effective_ev > base_threshold:
        # Scale bet fraction based on how much the effective EV exceeds the threshold
        # Target a 4% edge for a full 1.0 bet.
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0