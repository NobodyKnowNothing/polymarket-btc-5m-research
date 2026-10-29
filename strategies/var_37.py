# Var 37 | BT: -1.532 | Live PnL: $-7.66 | Trades: 60
# Desc: Gaussian-Anchor Microstructure Capture (GAMC) — Combines a Gaussian-derived "fair probability" (based on the exchange price's statistical distance from the strike) with the ML model's prediction to id
# Stats: {"live_returns": [-1.0, -0.3719632923603058, 0.08338749408721924, -0.3580634593963623, 0.20413166284561157, -1.0, -1.0, -0.549573540687561, 0.2454792857170105, 1.092555, 0.09182599999999996, 0.19025099999999995, -1.0, -1.0, -0.46403366327285767, 0.968532, 0.5945689999999999, -1.0, -1.0, 0.08073200000000003, -1.0, -1.0, 0.15118699999999996, -0.520043671131134, 0.021131962537765503, -0.28195473551750183, -1.0, 0.13876599999999994, -1.0, -1.0, 4.521505832672119, 3.1194439999999997, 0.4731161594390869, -0.26509079337120056, -1.0, 0.7611209999999999, -0.5951017141342163, 0.4265444278717041, 0.04383671283721924, 1.28551, -1.0, 1.092555, -1.0, -0.5639711022377014, -1.0, -0.3965955376625061, 0.4688166379928589, -1.0, -0.33538028597831726, 0.11776995658874512, -0.5202169418334961, 0.6356866359710693, 0.023511797189712524, -0.7634106874465942, 0.07831418514251709, -1.0, 0.8250329999999999, -1.0, -1.0, 0.5889719128608704], "live_pnl": -7.661111831665039, "live_wins": 27, "live_total": 60, "age": 123, "rounds_underwater": 60, "abstain_rounds": 1, "backtest_score": -1.5316816881684336, "backtest_acc": 0.5489897064430042, "backtest_n": 1784}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the statistical distance of the current price from the strike price
    and the synchronization between order flow and book imbalance.
    """
    df = df.copy()

    # 1. Price Z-Score: How many volatility units is the current price from the strike?
    # This provides a "structural" view of the market state.
    df['price_z_score'] = (df['binance_price'] - df['strike_price']) / (df['mean_volatility'] + 1e-6)

    # 2. Delta-OBI Synchronization: Are aggressive trades hitting the side 
    # with the most imbalance? (High positive = strong buying pressure)
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 3. Volatility-Normalized Flow: Aggressive flow relative to current volatility.
    df['vol_delta_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Velocity: Rate of change in Binance price.
    df['binance_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 12.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Gaussian-Anchor Microstructure Capture".
    It anchors the ML model's prediction to the statistical reality of the 
    current price position relative to the strike.
    """
    # 1. Calculate "Fair Probability" using the Gaussian CDF of the price Z-score
    # This represents the probability of the outcome based solely on current price/vol.
    z_score = features_row.get('price_z_score', 0.0)
    # math.erf(z / sqrt(2)) is the standard way to compute the Normal CDF
    fair_prob_up = 0.5 * (1 + math.erf(z_score / 1.41421356))

    # 2. Hybrid Probability: Blend the ML model with the structural fair probability
    # We give the ML model more weight (60%) but use the price as a reality anchor (40%).
    combined_prob_up = (0.6 * prob_up) + (0.4 * fair_prob_up)
    combined_prob_down = 1.0 - combined_prob_up

    # 3. Calculate Expected Value (EV)
    ev_up = combined_prob_up - cost_up
    ev_down = combined_prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 4. Dynamic Hurdle
    # Higher volatility requires a higher edge to overcome noise.
    vol = features_row.get('mean_volatility', 0.01)
    hurdle = 0.025 + (0.04 * vol)

    if best_ev < hurdle:
        return None, 0.0

    # 5. Microstructure Alignment Filter
    # We only take high-conviction trades if the aggressive flow (Delta) 
    # supports the direction of our trade.
    mean_delta = features_row.get('mean_delta', 0.0)
    multiplier = 1.0
    
    if side == 'Up' and mean_delta < 0:
        multiplier = 0.3  # Significant penalty for fighting aggressive sellers
    elif side == 'Down' and mean_delta > 0:
        multiplier = 0.3  # Significant penalty for fighting aggressive buyers

    # 6. Execution & Sizing
    # Scale by edge magnitude, microstructure alignment, and time remaining.
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))
    
    # The bet size is a function of how much the edge exceeds the hurdle.
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage/spread
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction