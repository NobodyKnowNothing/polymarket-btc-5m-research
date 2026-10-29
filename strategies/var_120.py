# Var 120 | BT: 0.442 | Live PnL: $-10.86 | Trades: 61
# Desc: Implements "Binance-Lead Impulse Scaling (BLIS)," which distinguishes between "Reinforcing Impulses" (where Binance's order flow and book imbalance amplify the aggregate market trend) and "Absorption/
# Stats: {"live_returns": [1.9791409999999998, -1.0, -1.0, -1.0, -0.9067234992980957, 6.089578151702881, 0.07195639610290527, -1.0, -1.0, -1.0, -1.0, 3.110865592956543, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.950496, -1.0, -0.9685525298118591, 2.280573, 1.8077519999999998, -1.0, -0.9234475493431091, -1.0, 0.8548961877822876, -1.0, -0.781250536441803, -1.0, -1.0, 2.651597, -1.0, -0.6983705163002014, -1.0, -1.0, 2.950496, -1.0, 1.3398933410644531, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.015964090824127197, -1.0, 0.05925399999999992, -1.0, -1.0, 0.09182599999999996, 1.4562249999999999, -1.0, -0.6754383444786072, 2.651597, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0], "live_pnl": -10.862264633178711, "live_wins": 17, "live_total": 61, "age": 143, "rounds_underwater": 34, "abstain_rounds": 0, "backtest_score": 0.4418433827364642, "backtest_acc": 0.564983888292159, "backtest_n": 2766}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence of the primary liquidity provider (Binance) from the market mean.
    """
    X = df.copy()

    # 1. Binance Excess Dynamics
    # Measures how much Binance is deviating from the cross-exchange average.
    # Positive = Binance is more aggressive/imbalanced than the market.
    X['binance_delta_excess'] = X['binance_delta'] - X['mean_delta']
    X['binance_obi_excess'] = X['binance_obi'] - X['mean_obi']

    # 2. Binance Impulse Signal
    # The product of excess delta and excess OBI. 
    # Positive = Binance is driving a trend (Reinforcement).
    # Negative = Binance is countering the trend (Absorption).
    X['binance_impulse'] = X['binance_delta_excess'] * X['binance_obi_excess']

    # 3. Relative Volatility
    # Normalizing price dispersion by volatility to identify high-conviction moves.
    X['dispersion_vol_ratio'] = X['price_std'] / (X['mean_volatility'] + 1e-6)

    # Drop metadata and target as per instructions
    # Note: seconds_remaining is passed to strategy_filter separately
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup: handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Binance-Lead Impulse Scaling (BLIS)".

    Logic:
    - We identify if Binance is 'Reinforcing' or 'Absorbing' the market trend.
    - Reinforcement: Binance delta and OBI both move in the same direction as the mean.
    - Absorption: Binance delta and OBI move against the mean (high friction).
    - We amplify the expected value (EV) and bet size during Reinforcement.
    - We heavily penalize EV and increase the hurdle during Absorption.
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
    impulse = features_row.get('binance_impulse', 0.0)
    vol = features_row.get('mean_volatility', 0.01)
    dispersion = features_row.get('price_std', 0.01)

    # 1. Calculate Normalized Impulse
    # We normalize the impulse by the product of volatility and dispersion 
    # to ensure we are measuring meaningful microstructure shifts.
    norm_impulse = impulse / (vol * dispersion + 1e-6)

    # 2. Determine Multiplier and Threshold based on Regime
    # Regime: Reinforcement (impulse > 0)
    if norm_impulse > 0:
        # Scale multiplier up to 2.0 for strong Binance-led momentum
        multiplier = 1.0 + min(1.0, 0.5 * abs(norm_impulse))
        threshold = 0.025 / multiplier
    # Regime: Absorption (impulse < 0)
    elif norm_impulse < 0:
        # Heavily penalize trades where Binance is fighting the trend
        multiplier = 0.4
        threshold = 0.05  # Higher hurdle for absorption
    # Regime: Neutral
    else:
        multiplier = 0.8
        threshold = 0.04

    # 3. Time-Gated Hurdle
    # Increase the required edge as the window closes to avoid late-stage noise.
    time_decay = 1.0 + (40.0 / (seconds_remaining + 1.0))
    adjusted_threshold = threshold * time_decay

    # 4. Apply Multiplier to EV and Decide
    effective_ev = best_ev * multiplier

    if effective_ev > adjusted_threshold:
        # 5. Dynamic Sizing
        # Scale bet size by the effective edge, capped at 1.0.
        # We use a target edge of 4% as the baseline for full sizing.
        bet_fraction = (effective_ev / 0.04) * multiplier
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0