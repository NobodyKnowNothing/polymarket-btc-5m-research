# Var 76 | BT: -0.381 | Live PnL: $-1.03 | Trades: 147
# Desc: Implements "Cross-Market Congruence Scaling (CMCS)," which uses a time-decayed alignment score between Perp funding and Spot microstructure to dynamically adjust EV thresholds and bet sizing.
# Stats: {"live_returns": [-1.0, -1.0, 1.3396370000000002, 0.06987699999999997, 0.11476700000000006, 1.6554160000000002, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, 4.245841, -1.0, 1.5191249999999998, 0.10316799999999993, -1.0, -1.0, 0.13876599999999994, 0.06987699999999997, 1.9791409999999998, 2.173394, 0.1769179999999999, 2.073133, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, 0.04885600000000001, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, 1.5854279999999998, 1.3964720000000002, 0.15118699999999996, 0.08073200000000003, 1.28551, -1.0, -1.0, 0.08073200000000003, -1.0, 1.049452, 1.233902, -1.0, -1.0, -1.0, 0.12663000000000002, -1.0, 1.1375739999999999, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, 3.505625, 1.1375739999999999, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, -1.0, 2.651597, 0.04885600000000001, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.05925399999999992, -1.0, -1.0, 1.092555, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.026145999999994, "live_wins": 56, "live_total": 147, "age": 202, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.3807189499970219, "backtest_acc": 0.5343199795866292, "backtest_n": 3235}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the alignment between Perp market sentiment (funding) and Spot market 
    microstructure (OBI and Delta).
    """
    df = df.copy()

    # 1. Cross-Market Alignment Signals
    # We want to know if the aggressive flow (Delta) is supported by the 
    # Perp market (Funding) and the Spot book (OBI).
    df['funding_delta_sign'] = np.sign(df['mean_funding']) * np.sign(df['mean_delta'])
    df['obi_delta_sign'] = np.sign(df['mean_obi']) * np.sign(df['mean_delta'])
    
    # 2. Volatility-Adjusted Flow
    # How much delta is moving per unit of volatility?
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price-to-Strike Dynamics
    # Distance from strike normalized by the percentage move
    df['dist_strike_norm'] = df['price_vs_strike_pct'] / (df['price_std'] + 1e-6)

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
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Market Congruence Scaling (CMCS)".

    Logic:
    - Calculate an 'alignment score' based on whether Funding and OBI agree with Delta.
    - Apply time-decay: Microstructure alignment matters more early in the window; 
      as the window closes, we rely more on the model's raw probability.
    - Dynamically adjust the required EV threshold and bet multiplier:
        - High Alignment (Trend): Lower EV threshold, higher bet size.
        - Low Alignment (Divergence/Exhaustion): Higher EV threshold, lower bet size.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 1. Calculate Alignment Score
    # f_d_sign and o_d_sign are pre-computed in engineer_features
    f_d_sign = features_row.get('funding_delta_sign', 0.0)
    o_d_sign = features_row.get('obi_delta_sign', 0.0)
    
    # Composite alignment: 40% Funding agreement, 60% OBI agreement
    alignment = (f_d_sign * 0.4) + (o_d_sign * 0.6)

    # 2. Apply Temporal Decay
    # As seconds_remaining -> 0, effective_alignment -> 0 (Neutral Regime)
    time_weight = seconds_remaining / 300.0
    effective_alignment = alignment * time_weight

    # 3. Dynamic Thresholding and Sizing
    # Base EV threshold is 0.035. 
    # If alignment is +1.0, threshold drops to 0.025.
    # If alignment is -1.0, threshold rises to 0.045.
    req_ev = 0.035 - (0.01 * effective_alignment)
    
    # Base multiplier is 1.0.
    # If alignment is +1.0, multiplier is 1.2.
    # If alignment is -1.0, multiplier is 0.8.
    multiplier = 1.0 + (0.2 * effective_alignment)

    # 4. Execution
    if best_ev > req_ev:
        # Scale bet by how much the EV exceeds the base threshold
        bet_fraction = multiplier * (best_ev / 0.035)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0