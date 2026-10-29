# Var 148 | BT: -0.790 | Live PnL: $-4.06 | Trades: 60
# Desc: Implements "Liquidity-Flow Sweep (LFS)," which targets high-conviction breakouts by identifying "vacuum" conditions where aggressive delta (flow) is moving against a thin order book (disagreement betw
# Stats: {"live_returns": [0.03779674999999999, -0.25, 0.34911800000000004, -0.25, -0.25, -0.25480079650878906, -0.05, -0.25, -0.25, 0.002004951238632202, 0.737624, -0.25, -0.25, -0.25, -0.25, -0.2628193497657776, 0.472712, -0.25, -0.25, 0.001933749999999998, -0.25, -0.25, -0.25, -0.25, 0.33490925000000005, 0.3213775, -0.25, 0.51828325, -0.25, -0.25, 0.66289925, -0.25, -0.16769635677337646, -0.25, -0.25, 0.51828325, -0.25, -0.25, -0.25, -0.14857853949069977, -0.07296941429376602, -0.25, -0.25, -0.25, -0.8435001373291016, -0.25, -0.25, 0.7798609999999999, -0.10381253808736801, 0.025606364011764526, -0.33796730637550354, -0.25, -0.25, -0.38923579454421997, 0.02449512481689453, 0.023269087076187134, 0.015107154846191406, 0.0028417184948921204, 1.7474782466888428, -0.25], "live_pnl": -4.055779457092285, "live_wins": 18, "live_total": 60, "age": 101, "rounds_underwater": 57, "abstain_rounds": 0, "backtest_score": -0.7901063160964521, "backtest_acc": 0.5446100251596672, "backtest_n": 4202}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'Vacuum Effect': aggressive flow hitting a thin order book.
    """
    df = df.copy()

    # 1. Volatility Encoding: Convert string regime to float
    df['vol_regime_val'] = df['mean_volatility'].map({'high': 1.0, 'low': 0.0}).fillna(0.5)

    # 2. Vacuum Up: Positive delta (buying) hitting a negative OBI (more asks than bids)
    # This represents a 'sweep' of the ask side.
    df['vacuum_up'] = df['mean_delta'] * (1.0 - df['mean_obi'])

    # 3. Vacuum Down: Negative delta (selling) hitting a positive OBI (more bids than asks)
    # This represents a 'sweep' of the bid side.
    df['vacuum_down'] = (-df['mean_delta']) * (1.0 + df['mean_obi'])

    # 4. Delta Intensity: Magnitude of the aggressive flow
    df['delta_intensity'] = df['mean_delta'].abs()

    # 5. Microstructure Tension: Product of spread and volatility
    df['market_tension'] = df['price_spread'] * df['vol_regime_val']

    # Drop metadata, target, and non-numeric columns for XGBoost
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'mean_volatility']
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
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Flow Sweep (LFS)" logic.

    Logic:
    1. Identify Sweeps:
       - Vacuum Up: High positive 'vacuum_up' indicates aggressive buying into a thin book.
       - Vacuum Down: High positive 'vacuum_down' indicates aggressive selling into a thin book.
    2. Conditional Thresholding:
       - If a sweep is detected, we lower the EV hurdle (0.02) because the microstructure 
         is primed for a directional move.
       - If no sweep is detected, we require a much higher EV hurdle (0.045) to ensure 
         we only take high-probability model predictions.
    3. Dynamic Sizing:
       - Scale bet size by the magnitude of the vacuum signal and the available edge.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    v_up = features_row.get('vacuum_up', 0.0)
    v_down = features_row.get('vacuum_down', 0.0)
    tension = features_row.get('market_tension', 0.0)

    side = None
    bet_fraction = 0.0

    # --- 1. Sweep Detection & Execution ---
    # We prioritize 'Sweep' signals as they represent high-momentum liquidity gaps.
    
    # Case A: Buy Sweep (Aggressive buying into thin asks)
    if v_up > 0.25 and ev_up > 0.02:
        side = 'Up'
        # Scale bet by vacuum strength, but penalize if market tension (vol/spread) is too high
        scaling = min(1.0, (v_up / 1.5) * (1.0 / (1.0 + tension)))
        bet_fraction = scaling * (ev_up / 0.05)

    # Case B: Sell Sweep (Aggressive selling into thin bids)
    elif v_down > 0.25 and ev_down > 0.02:
        side = 'Down'
        scaling = min(1.0, (v_down / 1.5) * (1.0 / (1.0 + tension)))
        bet_fraction = scaling * (ev_down / 0.05)

    # --- 2. Standard Model-Driven Execution ---
    # If no sweep is present, we only trade on very high-conviction model edges.
    elif ev_up > 0.045:
        side = 'Up'
        bet_fraction = 0.25  # Conservative flat bet
    elif ev_down > 0.045:
        side = 'Down'
        bet_fraction = 0.25  # Conservative flat bet

    # --- 3. Final Constraints ---
    if side:
        # Time-decay: Reduce exposure as the window closes to avoid last-second volatility
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        bet_fraction = min(1.0, max(0.05, bet_fraction * time_factor))
        return side, bet_fraction

    return None, 0.0