# Var 178 | BT: 0.826 | Live PnL: $-0.16 | Trades: 61
# Desc: Implements "Microstructure Alignment & Lead-Venue Validation (MALV)," which uses a normalized alignment score between Order Book Imbalance and Delta to gate model edge, while scaling bet size based on
# Stats: {"live_returns": [-1.0, 0.11476700000000006, -1.0, -1.0, 2.795066, -1.0, -1.0, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, 0.06739670038223267, 12.389175, -1.0, 0.17432481050491333, -1.0, -1.0, -1.0, -1.0, 2.280573, 1.3964720000000002, -1.0, 2.950496, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, 2.950496, -1.0, -1.0, 0.20391000000000004, 0.08856666088104248, 0.05925399999999992, -1.0, 1.233902, -0.5969866514205933, -1.0, 2.651597, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.173394, -1.0, -1.0, -0.8413349986076355, -1.0, 2.651597, -1.0, -1.0], "live_pnl": -0.16436195373535156, "live_wins": 20, "live_total": 61, "age": 270, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.8258733532671626, "backtest_acc": 0.5537447353964475, "backtest_n": 1368}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure alignment and exchange-specific leadership.
    """
    df = df.copy()

    # 1. Microstructure Alignment: Do the intent (OBI) and action (Delta) agree?
    # Range: [-1, 1]. Positive = aggressive buying into bids. Negative = aggressive selling into asks.
    df['micro_alignment'] = np.sign(df['mean_delta']) * df['mean_obi']

    # 2. Delta Intensity: Magnitude of flow normalized by price volatility.
    # High values indicate significant aggressive movement relative to recent noise.
    df['delta_intensity'] = df['mean_delta'].abs() / (df['price_std'] + 1e-6)

    # 3. Lead-Venue Divergence: How much is Binance leading or lagging the aggregate?
    # Positive = Binance is more aggressive than the mean.
    df['delta_divergence'] = df['binance_delta'] - df['mean_delta']
    df['obi_divergence'] = df['binance_obi'] - df['mean_obi']

    # 4. Volatility-Adjusted Flow: Net flow relative to the current volatility regime.
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to micro-noise."""
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
    Decide whether to trade using "Microstructure Alignment & Lead-Venue Validation (MALV)".

    Logic:
    1. Edge Identification: Calculate EV for both sides.
    2. Microstructure Gating: 
       - Only trade 'Up' if the model sees EV and the microstructure alignment is positive.
       - Only trade 'Down' if the model sees EV and the microstructure alignment is negative.
    3. Confidence Scaling:
       - Scale up if the alignment is strong (closer to 1 or -1).
       - Scale down if Binance (the lead venue) is diverging from the aggregate trend.
       - Scale down if the signal is approaching exhaustion (extreme delta intensity).
    4. Temporal Decay: Increase caution as the market window closes.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    alignment = features_row.get('micro_alignment', 0.0)
    delta_div = features_row.get('delta_divergence', 0.0)
    delta_int = features_row.get('delta_intensity', 0.0)

    side = None
    best_ev = 0.0
    
    # Thresholds
    ev_threshold = 0.032
    alignment_threshold = 0.15

    # 1. Directional Gating
    if ev_up > ev_down and ev_up > ev_threshold:
        if alignment > alignment_threshold:
            side = 'Up'
            best_ev = ev_up
    elif ev_down > ev_up and ev_down > ev_threshold:
        if alignment < -alignment_threshold:
            side = 'Down'
            best_ev = ev_down

    if side:
        # 2. Dynamic Sizing Logic
        multiplier = 1.0

        # A) Lead-Venue Check: Is Binance supporting the direction?
        # If we are going Up, we want Binance delta to be >= mean delta.
        if side == 'Up' and delta_div < 0:
            multiplier *= 0.7
        elif side == 'Down' and delta_div > 0:
            multiplier *= 0.7

        # B) Alignment Strength: Scale with the quality of the micro-signal
        # If alignment is 1.0, multiplier becomes 2.0. If 0.2, multiplier is 1.2.
        if side == 'Up':
            multiplier *= (1.0 + alignment)
        else:
            multiplier *= (1.0 - alignment)

        # C) Exhaustion Check: If delta intensity is extremely high, reduce size (potential reversal)
        if delta_int > 5.0:
            multiplier *= 0.5

        # D) Time Factor: Stay aggressive until the last 60 seconds
        time_factor = min(1.0, seconds_remaining / 60.0)

        # Calculate final fraction
        # Target 4% EV for a full bet (scaled by multiplier and time)
        bet_fraction = (best_ev / 0.04) * multiplier * time_factor
        
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0