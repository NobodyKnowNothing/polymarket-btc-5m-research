# Var 113 | BT: -0.106 | Live PnL: $-1.62 | Trades: 126
# Desc: Implements "Gamma-Sensitive Impulse Gating (GSIG)," which recognizes that the predictive power of order flow (Delta/OBI) is non-linearly amplified when the price is near the strike price (high "gamma"
# Stats: {"live_returns": [-1.0, -1.0, 0.09578955173492432, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -0.8596148490905762, -0.9675391316413879, 3.505625, -1.0, 1.28551, 1.7294070000000001, -1.0, -1.0, 2.795066, 1.890848, 1.890848, -1.0, -1.0, 0.16389999999999993, -1.0, 0.15118699999999996, -0.9353131055831909, 0.246969, 1.1938486099243164, 2.950496, 0.8472435474395752, -1.0, -1.0, 0.67470383644104, -1.0, -1.0, 3.9731249999999996, 0.09578955173492432, -0.5939649343490601, -1.0, -1.0, -0.6552011370658875, 0.10316799999999993, -1.0, -0.7068820595741272, 3.505625, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, 0.03600156307220459, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, 3.727685, 0.08073200000000003, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, 0.028705999999999898, -1.0, -1.0, 2.950496, -0.9470036625862122, 1.4562249999999999, -1.0, 3.9731249999999996, -1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -1.0, 4.893558, -1.0, 0.03600156307220459, 2.39541, -1.0, 0.10316799999999993, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, 0.08872324228286743, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5516842007637024, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.12663000000000002, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.6179707050323486, "live_wins": 45, "live_total": 126, "age": 185, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.10564417957554406, "backtest_acc": 0.5697009674582234, "backtest_n": 3889}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the non-linear sensitivity of order flow near the strike price.
    """
    df = df.copy()

    # 1. Proximity Factor: How close is the price to the strike?
    # As price_vs_strike_pct approaches 0, proximity_factor explodes.
    # This captures the "Gamma" effect where small moves near the strike 
    # have disproportionate impacts on the market outcome.
    df['proximity_factor'] = 1.0 / (df['price_vs_strike_pct'].abs() + 0.001)

    # 2. Order Flow Acceleration: The first derivative of delta and OBI.
    # Captures the "impulse" or "ignition" of a move.
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_accel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 3. Proximity-Weighted Signals: Scaling delta and OBI by their proximity to strike.
    # This allows the XGBoost model to learn that delta is more "meaningful" near the strike.
    df['delta_proximity'] = df['mean_delta'] * df['proximity_factor']
    df['obi_proximity'] = df['mean_obi'] * df['proximity_factor']
    
    # 4. Momentum-Volatility Interaction
    df['vol_norm_delta_accel'] = df['delta_accel'] / (df['mean_volatility'] + 1e-6)

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
    Decide whether to trade using "Gamma-Sensitive Impulse Gating (GSIG)".

    Logic:
    - Strike-Zone (High Proximity): Price is near the strike. Small delta/OBI 
      alignments are highly predictive. Lower EV threshold (0.02).
    - Outlier-Zone (Low Proximity): Price is far from the strike. We only 
      bet if we see massive 'Impulse' (high delta acceleration) that 
      aligns with the order book. Higher EV threshold (0.05).
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
    prox = features_row.get('proximity_factor', 1.0)
    d_accel = features_row.get('delta_accel', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    delta = features_row.get('mean_delta', 0.0)

    # 1. Determine Regime based on Proximity
    # High proximity (> 20) means we are very close to the strike price
    if prox > 20:
        regime = 'strike'
        req_ev = 0.020
        base_multiplier = 1.0
    # Low proximity (< 5) means we are far from the strike
    elif prox < 5:
        regime = 'outlier'
        req_ev = 0.050
        base_multiplier = 0.6
    else:
        regime = 'mid'
        req_ev = 0.035
        base_multiplier = 0.8

    # 2. Impulse/Alignment Refinement
    # If acceleration and OBI are aligned, it's a high-conviction impulse.
    # If they are opposing, it's absorption/friction.
    alignment = np.sign(d_accel) == np.sign(obi) if abs(d_accel) > 1e-5 else False
    
    if alignment:
        # Boost confidence in impulse moves
        multiplier = base_multiplier * 1.2
    else:
        # Penalize trades where flow and book are fighting
        multiplier = base_multiplier * 0.5

    # 3. Outlier-Regime Specific Constraint
    # In the outlier zone, we ONLY want to trade if there is significant acceleration.
    if regime == 'outlier' and abs(d_accel) < 0.01:
        return None, 0.0

    # 4. Time-Gating
    # As the market closes, increase the required hurdle to avoid noise.
    if seconds_remaining < 45:
        req_ev *= 1.3
        multiplier *= 0.7

    # 5. Final Execution
    if best_ev > req_ev:
        # Scale bet size: higher edge = higher fraction, but capped at 1.0
        # We use the req_ev as the baseline for the scaling
        bet_fraction = (best_ev / req_ev) * multiplier
        return side, min(1.0, bet_fraction)

    return None, 0.0