# Var 92 | BT: -0.442 | Live PnL: $-9.27 | Trades: 61
# Desc: Implements "Funding-Microstructure Squeeze Capture (FMSC-2)," which identifies "Squeeze" regimes where extreme funding rates (cost of carry) and aggressive order flow (Delta) are in opposition, creati
# Stats: {"live_returns": [-0.49569272994995117, -0.5000325788592046, -0.5086749868903407, -0.4925667643547058, -0.33819180727005005, -0.5055120022900605, -0.5090214151195855, 0.028994768857955933, 0.028259098529815674, -0.5007729558157165, -0.5008641517548945, -0.5056937146585544, -0.5020876159099306, 0.5931085348129272, -0.5163114924147643, -0.4349406063556671, -0.5164289512777165, -0.5084166644798728, 2.0070354186855153, 0.08595079183578491, 0.0644565224647522, 0.9296776148963614, 1.1325913200068263, -0.5213521856562691, -0.5376348114296309, -0.5557268867234982, -0.5202300760631763, 2.5381367923659335, -0.3536917269229889, 1.567821236982756, -0.5451781671878543, -0.500172998883701, -0.29211604595184326, -0.5037003208384169, -0.6247195803130159, -0.4904243052005768, -0.3370765745639801, -0.39935359358787537, -0.4284765422344208, 1.2013758984232887, 0.020790159702301025, 0.38584139943122864, -0.44455480575561523, -0.5010111045570821, 0.4151812791824341, 0.8885722160339355, -0.5152588387956011, -0.5093736876151902, -0.36176905035972595, -0.5199134217503957, -0.5015245594091313, -0.5040172666668431, 0.9924077873408109, -0.5242272844071912, -0.5445548892021179, -0.5083925422837923, -0.5115326389311053, -0.5198345488402222, -0.6320584777171618, -0.5675089056293927, -0.5396956904064188], "live_pnl": -9.270089100135422, "live_wins": 16, "live_total": 61, "age": 87, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.4423573747953261, "backtest_acc": 0.5498678202355203, "backtest_n": 3247}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the interaction between funding rates and price-impact-normalized delta.
    """
    df = df.copy()

    # 1. Delta Normalization: Scale delta by price volatility to understand its relative impact
    df['delta_norm'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 2. Squeeze Signal: A non-linear interaction of funding, imbalance, and normalized flow
    # High positive = potential Short Squeeze (Up); High negative = potential Long Squeeze (Down)
    # We use mean_funding * mean_obi * delta_norm
    df['squeeze_signal'] = df['mean_funding'] * df['mean_obi'] * df['delta_norm']

    # 3. Funding Intensity: The absolute magnitude of the funding rate (scaled for model visibility)
    df['funding_intensity'] = df['mean_funding'].abs() * 10000.0

    # 4. OBI-Delta Alignment: Do the book and the flow agree?
    df['obi_delta_alignment'] = np.sign(df['mean_obi']) * np.sign(df['delta_norm'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
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
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Microstructure Squeeze Capture (FMSC-2)".

    Logic:
    - Identifies "Squeeze" regimes:
        - Long Squeeze (Down): Funding is positive (longs pay), but Delta/OBI are negative.
        - Short Squeeze (Up): Funding is negative (shorts pay), but Delta/OBI are positive.
    - Calculates a 'Squeeze Strength' based on the magnitude of the funding and flow.
    - Amplifies the Effective EV if the model's predicted side matches the squeeze direction.
    - Penalizes the Effective EV if the model predicts against the squeeze direction.
    - Scales bet size based on the strength of the squeeze and the remaining time.
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
    funding = features_row.get('mean_funding', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    delta_norm = features_row.get('delta_norm', 0.0)
    
    # 1. Determine Squeeze Direction
    # A Long Squeeze (Price Down) occurs when funding is positive but flow is negative
    # A Short Squeeze (Price Up) occurs when funding is negative but flow is positive
    squeeze_dir = None
    if funding > 0 and delta_norm < 0:
        squeeze_dir = 'Down'
    elif funding < 0 and delta_norm > 0:
        squeeze_dir = 'Up'

    # 2. Calculate Squeeze Strength (capped to prevent extreme volatility)
    # Scaling factor 10000 to bring funding (e.g. 0.0001) into a usable range
    squeeze_strength = abs(funding * 10000.0 * obi * delta_norm)
    squeeze_strength = min(squeeze_strength, 0.5) 

    # 3. Calculate Effective EV
    # If the model agrees with the squeeze, boost the EV. If it disagrees, dampen it.
    if squeeze_dir is not None:
        if side == squeeze_dir:
            effective_ev = best_ev * (1.0 + squeeze_strength)
        else:
            effective_ev = best_ev * (1.0 - squeeze_strength)
    else:
        effective_ev = best_ev

    # 4. Dynamic Thresholding and Time Decay
    # Squeezes are most explosive early in the window; decay the impact as time runs out.
    time_factor = min(1.0, seconds_remaining / 300.0)
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    
    # Apply time decay to the squeeze boost
    effective_ev = effective_ev * (0.5 + 0.5 * time_factor)

    if effective_ev > base_threshold:
        # Scale bet fraction: Higher squeeze strength = higher conviction
        # Base bet is 0.5, scaled up to 1.0 by squeeze strength
        bet_fraction = 0.5 + (0.5 * squeeze_strength)
        # Also scale by how much the EV exceeds the threshold
        ev_scale = min(1.0, effective_ev / 0.06)
        
        return side, min(1.0, bet_fraction * ev_scale)

    return None, 0.0