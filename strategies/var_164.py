# Var 164 | BT: 0.524 | Live PnL: $-9.80 | Trades: 60
# Desc: Implements "Microstructure Pressure Gradient (MPG)," which distinguishes between "Intent-Driven Trends" (where order book imbalance velocity and aggressive flow velocity align) and "Aggressive Absorpt
# Stats: {"live_returns": [-1.0, 0.892141580581665, -1.0, -1.0, 0.13876599999999994, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.092555, -1.0, 1.3964720000000002, -1.0, -0.8575402498245239, -1.0, -1.0, 0.12663000000000002, 1.282610297203064, -1.0, -1.0, 0.12663000000000002, 0.10834240913391113, -1.0, -1.0, -1.0, -1.0, 1.890848, 0.04474538564682007, -1.0, 3.1194439999999997, -1.0, 0.10316799999999993, -1.0, -1.0, -1.0, -1.0, -0.7356118559837341, 1.233902, 0.12132018804550171, -1.0, -1.0, 1.4562249999999999, 0.038675000000000015, 0.04885600000000001, 4.245841, -1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, 1.1846420000000002, 0.15118699999999996, -1.0, 0.1496160626411438], "live_pnl": -9.795268058776855, "live_wins": 24, "live_total": 60, "age": 77, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 0.524092126593797, "backtest_acc": 0.567466964451889, "backtest_n": 4369}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    velocity and alignment of order book intent (OBI) and aggressive flow (Delta).
    """
    df = df.copy()

    # 1. Velocity Features: How fast are the signals changing within the market window?
    # We group by market_slug to ensure diffs are only calculated within the same window.
    group = df.groupby('market_slug')
    
    df['obi_vel'] = group['mean_obi'].diff().fillna(0)
    df['delta_vel'] = group['mean_delta'].diff().fillna(0)
    df['price_spread_vel'] = group['price_spread'].diff().fillna(0)

    # 2. Pressure Index: Do the changes in intent (OBI) and execution (Delta) align?
    # Positive = Alignment (Trend building); Negative = Divergence (Absorption/Exhaustion)
    df['pressure_index'] = df['obi_vel'] * df['delta_vel']
    
    # 3. Divergence Metric: The gap between flow velocity and intent velocity
    df['delta_obi_divergence'] = df['delta_vel'] - df['obi_vel']

    # 4. Volatility-Adjusted Flow: Magnitude of flow relative to current noise
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    # We keep pm_cost_up/down as they are critical for the model to learn market pricing
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Pressure Gradient (MPG)".

    Logic:
    1. Identify Pressure Regimes:
       - TREND: OBI velocity and Delta velocity have the same sign (Intent matches Execution).
       - ABSORPTION: Delta velocity is high, but OBI velocity is zero or opposite (Flow is being eaten).
       - NEUTRAL: Low velocity in both.
    2. Conviction Scaling:
       - Trend (Aligned) = 1.4x conviction.
       - Absorption (Divergent) = 0.4x conviction.
       - Neutral = 1.0x conviction.
    3. Execution:
       - Apply conviction to the Expected Value (EV).
       - Scale bet size by edge, conviction, and a time-safety decay.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Determine best side
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    obi_vel = features_row.get('obi_vel', 0.0)
    delta_vel = features_row.get('delta_vel', 0.0)

    # 1. Regime Identification via Pressure Alignment
    # sign(obi_vel) * sign(delta_vel) will be 1 if aligned, -1 if diverging, 0 if one is flat
    alignment = np.sign(obi_vel) * np.sign(delta_vel)
    
    conviction = 1.0

    if alignment > 0:
        # Trend: Intent and Execution are moving together
        conviction = 1.4
    elif alignment < 0:
        # Divergence: Check if it's aggressive absorption
        # If delta is moving much faster than OBI, it's likely being absorbed by the book
        if abs(delta_vel) > abs(obi_vel) * 2.5:
            conviction = 0.4
        else:
            conviction = 0.7
    
    # 2. Adjusted EV Thresholding
    # We use a baseline hurdle of 0.02. Conviction scales the 'perceived' edge.
    effective_ev = best_ev * conviction

    if effective_ev > 0.02:
        # 3. Dynamic Bet Sizing
        # Target edge of 3.5% (0.035) for full sizing
        # Scale by: (Edge / Target) * Conviction * Time_Safety
        
        # Time Safety: Reduce bet size significantly if we are in the final 30 seconds 
        # to avoid "last-second" noise/volatility spikes.
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = (best_ev / 0.035) * conviction * time_factor

        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0