# Var 90 | BT: 0.026 | Live PnL: $-11.88 | Trades: 61
# Desc: Implements "Liquidity-Vacuum/Absorption Scaling (LVAS)," which differentiates between "Vacuum" regimes (aggressive flow meeting no liquidity, leading to breakouts) and "Absorption" regimes (aggressive
# Stats: {"live_returns": [-1.0, 1.1846420000000002, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.049452, -1.0, 0.038675000000000015, 0.13876599999999994, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.05925399999999992, -1.0, -1.0, 1.092555, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, -1.0], "live_pnl": -11.882747, "live_wins": 19, "live_total": 61, "age": 116, "rounds_underwater": 57, "abstain_rounds": 0, "backtest_score": 0.026338546798396922, "backtest_acc": 0.5668851663025006, "backtest_n": 3065}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    relationship between aggressive flow (Delta) and the existing liquidity (OBI).
    """
    df = df.copy()

    # 1. Delta-OBI Sign Match: 
    #  1.0: Same sign (Aggressive flow is meeting liquidity - Absorption)
    # -1.0: Opposite sign (Aggressive flow is hitting empty/opposite side - Vacuum)
    #  0.0: One of them is zero (Neutral)
    df['delta_obi_match'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 2. Vacuum Intensity: 
    # How much delta is moving through the book relative to the imbalance.
    # High value = Flow is "unsupported" by the book (potential breakout).
    # Low value = Flow is "heavily supported" (potential absorption).
    # We use log1p to dampen the impact of extreme outliers.
    df['vacuum_intensity'] = np.log1p(df['mean_delta'].abs() / (df['mean_obi'].abs() + 1e-6))

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
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Vacuum/Absorption Scaling (LVAS)".

    Logic:
    - Identify the regime: Vacuum (breakout potential) vs Absorption (reversal potential).
    - Vacuum Regime (match < 0): Aggressive flow is moving into a vacuum. 
      Multiply EV to capture high-conviction breakouts.
    - Absorption Regime (match > 0): Aggressive flow is being absorbed by a wall.
      Reduce conviction/bet size as the direction is likely to stall or reverse.
    - Scale bet size based on the 'vacuum_intensity' in breakout regimes.
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
    match = features_row.get('delta_obi_match', 0.0)
    intensity = features_row.get('vacuum_intensity', 0.0)

    # 1. Determine Regime Multiplier
    multiplier = 1.0
    if match < 0:
        # Vacuum Regime: Scale multiplier between 1.2 and 2.0 based on intensity
        # intensity/3.0 is a heuristic to normalize the log-scale intensity
        multiplier = 1.2 + (0.8 * min(1.0, intensity / 3.0))
    elif match > 0:
        # Absorption Regime: Significant penalty to conviction
        multiplier = 0.5
    
    # 2. Calculate Effective EV
    effective_ev = best_ev * multiplier

    # 3. Dynamic Thresholding
    # Require higher edge as the window closes to avoid volatility spikes
    base_threshold = 0.03
    if seconds_remaining < 100:
        base_threshold = 0.05
    elif seconds_remaining < 200:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # Scale bet fraction: Aim for 1.0 bet when effective_ev is ~4%
        # We use the multiplier to ensure we bet more on 'clean' vacuum moves
        bet_fraction = (best_ev / 0.04) * multiplier
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0