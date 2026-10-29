# Var 95 | BT: -0.532 | Live PnL: $-11.88 | Trades: 61
# Desc: Implements "Cross-Exchange Velocity & Flow Acceleration (CEVFA)," which scales conviction by detecting when Binance price velocity leads other exchanges and is corroborated by accelerating order flow
# Stats: {"live_returns": [0.12663000000000002, -0.4074573814868927, -1.0, -1.0, -1.0, 3.9731249999999996, 0.10316799999999993, 1.28551, -1.0, 2.39541, 1.28551, -1.0, -1.0, -1.0, -1.0, -0.7635188102722168, 2.280573, 0.15118699999999996, 0.032052457332611084, 0.10494285821914673, 0.07275736331939697, -1.0, -1.0, 0.11476700000000006, -1.0, -0.32596588134765625, -1.0, -1.0, 0.13876599999999994, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, 0.07697939872741699, -1.0, -1.0, -1.0, -1.0, -0.6186149716377258, -1.0, -1.0, -1.0, 2.39541, 0.041006624698638916, 0.5509371757507324, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -11.881368637084961, "live_wins": 21, "live_total": 61, "age": 75, "rounds_underwater": 24, "abstain_rounds": 0, "backtest_score": -0.5315710279474799, "backtest_acc": 0.5414377836159542, "backtest_n": 3589}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange lead-lag dynamics and order flow acceleration.
    """
    df = df.copy()

    # 1. Binance Velocity (Normalized by Volatility)
    # Captures how fast the lead exchange is moving relative to recent volatility
    df['binance_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['binance_velocity_norm'] = df['binance_velocity'] / (df['mean_volatility'] + 1e-6)

    # 2. Cross-Exchange Lag (Percentage)
    # Measures how far Binance has moved ahead of the Coinbase/Kraken average
    other_avg_price = (df['coinbase_price'] + df['kraken_price']) / 2
    df['lag_spread_pct'] = (df['binance_price'] - other_avg_price) / df['binance_price']

    # 3. Order Flow Acceleration
    # Captures the second derivative of aggressive delta (is buying/selling speeding up?)
    df['delta_acceleration'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 4. OBI Acceleration
    # Captures shifts in order book pressure momentum
    df['obi_acceleration'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Exchange Velocity & Flow Acceleration (CEVFA)".

    Logic:
    - Base Decision: Use model-predicted EV.
    - Conviction Boost: Increase bet size if Binance is leading the price (lag_spread) 
      AND the movement is accompanied by accelerating delta/OBI.
    - Conviction Penalty: Reduce bet size if the model predicts a direction but 
      microstructure (velocity/flow) is decelerating or moving against the prediction.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        direction = 1
    else:
        best_ev = ev_down
        side = 'Down'
        direction = -1

    # Minimum EV threshold to consider any trade
    if best_ev < 0.03:
        return None, 0.0

    # Extract engineered features
    vel_norm = features_row.get('binance_velocity_norm', 0.0)
    lag_pct = features_row.get('lag_spread_pct', 0.0)
    delta_accel = features_row.get('delta_acceleration', 0.0)
    obi_accel = features_row.get('obi_acceleration', 0.0)

    # 1. Signal Detection
    # Is Binance leading the move in our predicted direction?
    is_leading = (direction * vel_norm > 0) and (direction * lag_pct > 0)
    
    # Is the aggressive flow/book pressure accelerating in our direction?
    is_accelerating = (direction * delta_accel > 0) or (direction * obi_accel > 0)

    # 2. Multiplier Logic
    # Start with a base multiplier scaled by the magnitude of the edge
    multiplier = 0.5 

    if is_leading and is_accelerating:
        # High conviction: Momentum is confirmed by both price lead and flow acceleration
        multiplier = 1.0
    elif is_leading:
        # Moderate conviction: Price is leading, but flow hasn't caught up yet
        multiplier = 0.8
    elif is_accelerating:
        # Moderate conviction: Flow is accelerating, but price hasn't broken out yet
        multiplier = 0.7
    else:
        # Low conviction: Model predicts direction, but microstructure is stagnant or diverging
        multiplier = 0.3

    # 3. Time-Gating
    # As the window closes, the importance of "catch-up" (lag) increases, 
    # but the risk of noise also increases.
    if seconds_remaining < 60:
        multiplier *= 0.8

    # 4. Final Bet Sizing
    # Scale the multiplier by how much the EV exceeds our baseline 4% target
    bet_fraction = multiplier * (best_ev / 0.04)
    
    return side, min(1.0, max(0.0, bet_fraction))