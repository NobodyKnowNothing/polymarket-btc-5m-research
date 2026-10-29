# Var 106 | BT: 1.025 | Live PnL: $-5.84 | Trades: 60
# Desc: Implements "Microstructure Acceleration & Impact Scaling (MAIS)," which scales conviction based on whether aggressive order flow is accelerating (ignition) or decelerating (exhaustion) relative to the
# Stats: {"live_returns": [-1.0, -0.3799353837966919, -1.0, -0.9231430888175964, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 1.233902, -1.0, 0.8321251273155212, -1.0, -0.6410024166107178, -1.0, -1.0, -0.6188255548477173, -1.0, -1.0, -0.7781772613525391, 3.1194439999999997, -1.0, -1.0, -1.0, 0.04885600000000001, 1.0916364192962646, 0.07596564292907715, -1.0, 1.4562249999999999, 1.233902, 0.7659522294998169, 0.04885600000000001, -1.0, 2.795066, 0.19025099999999995, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, 4.245841, -1.0, -1.0, 1.1375739999999999, -1.0, 0.42417725920677185, -1.0, -1.0, 1.3964720000000002, 2.950496, 1.3396370000000002, 1.7294070000000001, -1.0, -0.38416603207588196, -1.0], "live_pnl": -5.836137771606445, "live_wins": 22, "live_total": 60, "age": 317, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 1.0249045779098314, "backtest_acc": 0.5756478658536586, "backtest_n": 1427}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding 
    first-order derivatives (acceleration) of flow and price.
    """
    df = df.copy()

    # 1. Delta Congruence: Agreement of aggressive flow across exchanges
    df['delta_congruence'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    ) / 3.0

    # 2. OBI Congruence: Agreement of book imbalance across exchanges
    df['obi_congruence'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    ) / 3.0

    # 3. Microstructure Acceleration: The change in delta flow (momentum of flow)
    # We group by market_slug to ensure diffs don't leak between windows
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 4. Price Velocity: The change in mid-price
    df['price_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata and target as per instructions
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup to handle NaNs from diff() and any potential infinities
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to handle noisy acceleration features."""
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
    Decide whether to trade using "Microstructure Acceleration & Impact Scaling (MAIS)".

    Logic:
    - Base Decision: Use the model's predicted EV.
    - Congruence Gate: Only trade if exchange delta agreement is high (|congruence| >= 0.66).
    - Acceleration Multiplier:
        - Ignition: Flow is moving in the predicted direction AND accelerating -> Increase bet.
        - Exhaustion: Flow is moving in the predicted direction BUT decelerating -> Decrease bet.
        - Absorption: Flow is moving AGAINST the predicted direction BUT decelerating (recovering) -> Moderate bet.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Threshold for minimum edge
    if best_ev < 0.025:
        return None, 0.0

    # Extract engineered features from the row
    congruence = features_row.get('delta_congruence', 0.0)
    delta_accel = features_row.get('delta_accel', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)

    # 1. Congruence Gate: Avoid trading when exchanges are in conflict
    if abs(congruence) < 0.66:
        return None, 0.0

    # 2. Calculate Acceleration Multiplier
    # multiplier > 1.0 = High Conviction, < 1.0 = Low Conviction
    multiplier = 1.0

    if side == 'Up':
        if mean_delta > 0:
            if delta_accel > 0:
                multiplier = 1.30  # Ignition: Buying is accelerating
            else:
                multiplier = 0.70  # Exhaustion: Buying is slowing down
        else:
            if delta_accel > 0:
                multiplier = 0.85  # Absorption: Selling is slowing/recovering
            else:
                multiplier = 0.60  # Aggressive Selling: Momentum against us
    else: # side == 'Down'
        if mean_delta < 0:
            if delta_accel < 0:
                multiplier = 1.30  # Ignition: Selling is accelerating
            else:
                multiplier = 0.70  # Exhaustion: Selling is slowing down
        else:
            if delta_accel < 0:
                multiplier = 0.85  # Absorption: Buying is slowing/recovering
            else:
                multiplier = 0.60  # Aggressive Buying: Momentum against us

    # 3. Time-Gating: Reduce bet size if we are very close to the close to avoid noise
    if seconds_remaining < 45:
        multiplier *= 0.7

    # 4. Final Execution
    # Scale bet size based on the magnitude of the edge relative to a 4% target
    bet_fraction = multiplier * (best_ev / 0.04)
    
    return side, min(1.0, max(0.0, bet_fraction))