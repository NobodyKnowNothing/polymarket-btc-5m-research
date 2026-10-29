# Var 130 | BT: -0.512 | Live PnL: $-1.39 | Trades: 61
# Desc: Implements "Temporal-Volatility Adaptive Gating (TVAG)," which replaces multiplicative scaling with an additive dynamic threshold that raises the entry bar as time decays or volatility increases, whil
# Stats: {"live_returns": [-1.0, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, 0.12663000000000002, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, -1.0, -1.0, -1.0, 2.39541, 1.1375739999999999, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, 1.3396370000000002, 1.3396370000000002, -1.0, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, 3.727685, -1.0, -1.0, 2.39541, -1.0], "live_pnl": -1.3873170000000004, "live_wins": 21, "live_total": 61, "age": 101, "rounds_underwater": 52, "abstain_rounds": 0, "backtest_score": -0.5121186745002497, "backtest_acc": 0.5581064607882974, "backtest_n": 3137}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure alignment and price velocity derivatives.
    """
    df = df.copy()

    # 1. Price Velocity: Rate of price change relative to distance from strike.
    # We add 1.0 to denominator to prevent division by zero/inf as seconds_remaining -> 0.
    df['price_velocity'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1.0)

    # 2. Velocity Acceleration: The second derivative of price movement.
    # Group by market_slug to prevent leaking information across different 5-min windows.
    df['velocity_accel'] = df.groupby('market_slug')['price_velocity'].diff().fillna(0)

    # 3. Delta-OBI Alignment: Do aggressive trades and the limit order book agree?
    # 1.0 = Agreement (both bullish or both bearish), -1.0 = Clash (absorption/divergence), 0.0 = Neutral.
    df['delta_obi_alignment'] = np.sign(df['sum_delta']) * np.sign(df['mean_obi'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle potential infs from velocity calculations and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signal capture."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Temporal-Volatility Adaptive Gating (TVAG)".

    Logic:
    - Instead of scaling EV, we dynamically adjust the MINIMUM required EV threshold.
    - Threshold increases as:
        a) Time runs out (Information Decay): Less time for the edge to manifest.
        b) Volatility increases (Risk): Higher noise requires higher conviction.
    - Conviction Multiplier: Boosts effective EV if Delta and OBI are aligned (congruence).
    - Bet Sizing: Scales based on the magnitude of the effective edge.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 1. Extract features
    alignment = features_row.get('delta_obi_alignment', 0.0)
    volatility = features_row.get('mean_volatility', 0.0)
    
    # 2. Dynamic Threshold Calculation
    # Base threshold is 3%. We add penalties for low time and high volatility.
    time_factor = max(0.0, seconds_remaining / 300.0)  # 1.0 at start, 0.0 at end
    
    # As time_factor -> 0, threshold increases by up to 5%
    # As volatility increases, threshold increases proportionally
    dynamic_threshold = 0.03 + (1.0 - time_factor) * 0.05 + (volatility * 0.05)

    # 3. Conviction Multiplier
    # If alignment is 1.0 (congruent), conviction is 1.3x.
    # If alignment is -1.0 (clash), conviction is 0.7x.
    conviction = 1.0 + (0.3 * alignment)
    
    effective_ev = best_ev * conviction

    # 4. Execution
    if effective_ev > dynamic_threshold:
        # Target a 4% effective edge for a full bet, scaling linearly.
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0