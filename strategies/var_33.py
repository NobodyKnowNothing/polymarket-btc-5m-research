# Var 33 | BT: -0.672 | Live PnL: $-2.93 | Trades: 62
# Desc: Aggressive Sweep Capture (ASC) — Targets high-velocity price movements by identifying "liquidity sweeps" where aggressive delta acts against a thin/imbalanced order book, creating a vacuum effect.
# Stats: {"live_returns": [1.8077519999999998, -1.0, 1.8077519999999998, 0.04661789536476135, 1.890848, -1.0, -1.0, -1.0, -1.0, -0.9625730514526367, -0.9548549056053162, -1.0, -1.0, 1.721121072769165, -1.0, -0.29151085019111633, -1.0, 0.19025099999999995, -1.0, 0.017811745405197144, 2.073133, 5.726361, -1.0, 1.890848, -1.0, -1.0, 1.6554160000000002, 3.727685, 4.245841, -1.0, -1.0, -1.0, -0.382715106010437, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -0.5647274255752563, -1.0, 0.0668877363204956, -1.0, -1.0, -1.0, -0.7916132211685181, -1.0, -0.8078189492225647, 0.13876599999999994, 0.35783180594444275, -1.0, 4.550646, 3.1194439999999997, -0.8548066020011902, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7356094121932983], "live_pnl": -2.93388032913208, "live_wins": 20, "live_total": 62, "age": 83, "rounds_underwater": 4, "abstain_rounds": 0, "backtest_score": -0.6716631052527006, "backtest_acc": 0.5684045472363779, "backtest_n": 2035}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    acceleration metrics and the 'sweep' potential (delta vs OBI divergence).
    """
    df = df.copy()

    # 1. Velocity and Acceleration
    # Captures the rate of change in order flow and price
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_vel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['price_accel'] = df.groupby('market_slug')['binance_price'].diff().diff().fillna(0)

    # 2. Funding-Delta Interaction
    # Measures if aggressive flow is supported or countered by perp funding sentiment
    df['funding_delta_prod'] = df['mean_funding'] * df['mean_delta']

    # 3. Sweep Potential (Mathematical proxy)
    # High when delta is high but OBI is low/opposing (indicating liquidity is being eaten)
    df['sweep_potential'] = abs(df['mean_delta']) * (1 - abs(df['mean_obi']))

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters tuned for capturing non-linear acceleration signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Aggressive Sweep Capture".
    The strategy identifies 'sweeps'—moments where aggressive delta is hitting 
    a side of the book that is heavily imbalanced in the opposite direction, 
    signaling a high-probability liquidity vacuum move.
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract Microstructure Features
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)

    # 3. Identify "Sweep" Condition
    # A sweep occurs when aggressive flow (Delta) is moving against the book imbalance (OBI).
    # If side is 'Up', we want positive delta (buying) but negative OBI (sell-side heavy).
    # If side is 'Down', we want negative delta (selling) but positive OBI (buy-side heavy).
    is_sweep = False
    if side == 'Up' and mean_delta > 0 and mean_obi < 0:
        is_sweep = True
    elif side == 'Down' and mean_delta < 0 and mean_obi > 0:
        is_sweep = True

    # 4. Dynamic Hurdle & Multiplier
    # We require a higher EV for "grind" trades and reward "sweep" trades.
    hurdle = 0.035
    
    if best_ev < hurdle:
        return None, 0.0

    # Multiplier: Heavily weight sweeps, penalize non-sweep momentum (which is prone to reversal)
    multiplier = 1.8 if is_sweep else 0.4

    # 5. Execution & Sizing
    # Time-decay: Use a square root decay to keep exposure higher in the mid-window 
    # but drop off sharply as the 5-min window closes.
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # Final bet fraction: Scale by (Edge/Hurdle) * Microstructure Confidence * Time
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction