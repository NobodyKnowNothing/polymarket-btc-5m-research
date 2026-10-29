# Var 109 | BT: -0.523 | Live PnL: $-0.10 | Trades: 64
# Desc: Implements "Confluence-Gated Dislocation Capture (CGDC)," which targets mispriced tokens by requiring high alignment between the ML model, order book imbalance (OBI), and price momentum, while applyin
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, 2.0316429138183594, -1.0, 3.505625, -1.0, -1.0, -1.0, -0.6932414770126343, -1.0, 1.636513113975525, -1.0, -1.0, -1.0, -0.9462416768074036, -1.0, 1.8077519999999998, -1.0, -1.0, 0.038675000000000015, 0.038675000000000015, 2.950496, 1.7645650000000002, -1.0, 0.04826551675796509, 0.7284088134765625, -1.0, -1.0, -1.0, 0.11476700000000006, 1.7049177885055542, 2.651597, -1.0, -1.0, -1.0, 0.04885600000000001, -1.0, -1.0, -1.0, 1.5854279999999998, -0.9918227791786194, -0.6092579960823059, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 2.795066, 3.727685, -0.824240505695343, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -0.10107994079589844, "live_wins": 22, "live_total": 64, "age": 248, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": -0.5231820817651067, "backtest_acc": 0.5715557533941688, "backtest_n": 1796}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the confluence of order flow, momentum, and volatility-normalized delta.
    """
    df = df.copy()

    # 1. Confluence Features: Do OBI and Momentum agree with the aggressive Delta?
    # Positive means OBI/Momentum and Delta are pushing in the same direction.
    df['obi_delta_confluence'] = df['mean_obi'] * np.sign(df['mean_delta'])
    df['momentum_delta_confluence'] = df['mean_momentum'] * np.sign(df['mean_delta'])

    # 2. Volatility-Normalized Delta: Is the flow significant relative to current volatility?
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price Velocity: The raw change in Binance price
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

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
    Decide whether to trade using "Confluence-Gated Dislocation Capture (CGDC)".

    Logic:
    - Identifies the side with the highest Expected Value (EV).
    - Calculates 'Agreement': How well OBI and Momentum align with the model's direction.
    - Calculates 'Consensus': How closely exchange prices agree (inverse of spread).
    - Applies a 'Trap Penalty': If the model and order flow disagree, the required EV threshold spikes.
    - Scales bet size by the magnitude of the edge, the strength of agreement, and market consensus.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        ev = ev_up
        side = 'Up'
        direction = 1
    else:
        ev = ev_down
        side = 'Down'
        direction = -1

    # 1. Extract engineered signals
    # Agreement: 1.0 if flow matches model, 0.0 if neutral, -1.0 if flow opposes model
    obi_conf = features_row.get('obi_delta_confluence', 0.0)
    mom_conf = features_row.get('momentum_delta_confluence', 0.0)
    agreement = ((obi_conf * direction) + (mom_conf * direction)) / 2.0
    
    # Consensus: 1.0 if prices are identical, approaching 0.0 as spread increases
    price_spread = features_row.get('price_spread', 0.0)
    binance_price = features_row.get('binance_price', 1.0)
    consensus = max(0.0, 1.0 - (price_spread / binance_price))

    # 2. Dynamic Thresholding
    # Base threshold starts at 3% edge, adjusted by volatility
    volatility = features_row.get('mean_volatility', 0.01)
    threshold = 0.03 + (volatility * 2.0)

    # Trap Penalty: If order flow/momentum disagrees with the model, require much higher edge
    if agreement < 0:
        threshold += 0.05

    # Time-Gating: Tighten requirements as the window closes to avoid noise
    if seconds_remaining < 60:
        threshold *= 1.5

    # 3. Execution and Sizing
    if ev > threshold:
        # Scale bet size by:
        # - Magnitude of edge relative to threshold
        # - Strength of signal agreement (0.5 to 1.0 multiplier)
        # - Strength of cross-exchange consensus (0.5 to 1.0 multiplier)
        agreement_mult = 0.5 + (0.5 * max(0, agreement))
        consensus_mult = 0.5 + (0.5 * consensus)
        
        bet_fraction = (ev / threshold) * agreement_mult * consensus_mult
        return side, min(1.0, bet_fraction)

    return None, 0.0