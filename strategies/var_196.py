# Var 196 | BT: 0.562 | Live PnL: $-3.16 | Trades: 36
# Desc: Implements Volatility-Adjusted Signal-to-Noise (VASN), which switches between OBI-stability and Delta-efficiency as the primary signal based on volatility, while scaling the entry hurdle and bet size
# Stats: {"live_returns": [2.39541, -0.9959602453566792, 1.7251917905504686, -0.971314378345172, 0.02835855850031954, -0.996516903161011, 0.02827821358108351, -0.9990665706619573, -0.9987593338839306, 0.12466258262514895, 1.8868109846973322, 0.17578400469938316, 0.11383305436035962, 1.2326719210786208, 0.0917324942366382, 0.16197239079707482, 0.03817778193906585, -0.995962214737759, -0.9945312253561621, 2.385879356245404, -0.9973448770390755, -0.9990329599154709, -0.9948759961130216, -0.9978347688014219, -0.98810300539152, 1.6507593156160838, 1.4532822753606718, 0.028172864049545332, 0.17673130075129306, -0.9951266696542747, 1.0445144865343994, -0.9972478690868771, -0.9940151005508171, -0.9931185605120872, -0.9984233546314433, -0.9936626245288136], "live_pnl": -3.158673282104602, "live_wins": 18, "live_total": 36, "age": 263, "rounds_underwater": 4, "abstain_rounds": 59, "backtest_score": 0.5623291260781289, "backtest_acc": 0.5588130774697939, "backtest_n": 958}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    signal-to-noise ratios (Signal / Volatility).
    """
    X = df.copy()

    # 1. Signal-to-Noise Ratios
    # We normalize the strength of intent (OBI) and action (Delta) by the current volatility.
    # This helps the model distinguish between 'meaningful' flow and 'noisy' flow.
    X['delta_efficiency'] = X['mean_delta'] / (X['mean_volatility'] + 1e-6)
    X['obi_stability'] = X['mean_obi'] / (X['mean_volatility'] + 1e-6)

    # 2. Price-Flow Divergence (Efficiency of delta in moving price)
    X['price_delta_ratio'] = X['price_vs_strike_pct'] / (X['mean_delta'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

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
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Adjusted Signal-to-Noise (VASN)".

    Logic:
    1. Regime Detection:
       - High Volatility: Market is driven by aggressive flow (Delta). We use 'delta_efficiency'.
       - Low Volatility: Market is driven by order book structure (OBI). We use 'obi_stability'.
    2. Dynamic Hurdle:
       - The required Expected Value (EV) increases as volatility increases to avoid whipsaws.
    3. Signal-to-Noise Bet Sizing:
       - Bet size is scaled by the ratio of the signal strength to the volatility.
       - We apply a penalty if the price is already significantly extended from the strike.
    """
    # Exit if too little time remains
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    vol = features_row.get('mean_volatility', 0.0)
    delta_eff = features_row.get('delta_efficiency', 0.0)
    obi_stab = features_row.get('obi_stability', 0.0)
    price_dist = features_row.get('price_vs_strike_pct', 0.0)

    # 1. Dynamic Hurdle: Higher vol requires higher edge
    # If vol is 0.002 (0.2%), hurdle is ~0.035. If vol is 0.01 (1%), hurdle is ~0.075.
    hurdle = 0.025 + (vol * 5.0)

    # 2. Regime-Based Signal Selection
    # High vol (> 0.1%) -> Delta-driven. Low vol -> OBI-driven.
    if vol > 0.001:
        signal = delta_eff
    else:
        signal = obi_stab

    side = None
    bet_fraction = 0.0

    # 3. Execution Logic
    if ev_up > hurdle and signal > 0:
        side = 'Up'
    elif ev_down > hurdle and signal < 0:
        side = 'Down'

    if side:
        # 4. Signal-to-Noise Scaling
        # bet_fraction = 1 / (1 + Noise/Signal). 
        # If signal is high and vol is low, bet is large.
        snr_scale = 1.0 / (1.0 + (vol / (abs(signal) + 1e-6)))
        
        # 5. Extension Penalty
        # Avoid buying the extreme ends of a 5-min candle.
        extension_penalty = 1.0 / (1.0 + abs(price_dist) * 15)
        
        bet_fraction = snr_scale * extension_penalty
        
        # Clamp bet size
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0