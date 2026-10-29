# Var 73 | BT: 0.047 | Live PnL: $-11.61 | Trades: 61
# Desc: Implements "Microstructure Force Convergence (MFC)," which scales conviction based on the alignment between current order flow/book pressure and the acceleration of that pressure.
# Stats: {"live_returns": [-1.0, -0.20052669942378998, 2.950496, -1.0, -1.0, -1.0, -1.0, 1.6554160000000002, 2.39541, -1.0, -1.0, -1.0, -0.5708502531051636, -1.0, -1.0, 1.5848238468170166, -0.6929261684417725, 1.1846420000000002, 0.6016606092453003, -0.2067042738199234, 1.2483329772949219, 1.978393316268921, -1.0, -1.0, -0.3611419200897217, -1.0, -1.0, -0.4620887041091919, 0.13876599999999994, 0.32890862226486206, -0.9485757946968079, -0.8948637247085571, 3.1194439999999997, -0.7033917903900146, -1.0, -1.0, -1.0, -0.35067594051361084, -0.29349106550216675, 2.39541, 0.019887685775756836, 0.29921412467956543, -1.0, -1.0, 1.092555, 1.0182404518127441, -1.0, -1.0, -0.29349106550216675, -1.0, -1.0, -0.9672730565071106, -1.0, 1.670377492904663, -0.32916900515556335, -0.6992329359054565, -0.34042856097221375, -1.0, -1.0, 0.021667346358299255, -1.0], "live_pnl": -11.611183166503906, "live_wins": 18, "live_total": 61, "age": 86, "rounds_underwater": 38, "abstain_rounds": 0, "backtest_score": 0.047120793357538715, "backtest_acc": 0.5521393799641302, "backtest_n": 3072}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'Microstructure Force'—the product of flow, liquidity, and their accelerations.
    """
    df = df.copy()

    # 1. Calculate first-order derivatives (velocities/accelerations)
    # Grouping by market_slug is critical to prevent cross-market leakage.
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_accel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Microstructure Force: A measure of 'pressure'.
    # High force occurs when current delta/OBI and their recent changes are aligned.
    # This captures both the state (level) and the momentum (acceleration).
    df['micro_force'] = (df['mean_delta'] * df['mean_obi']) + (df['delta_accel'] * df['obi_accel'])

    # 3. Volatility-Normalized Force: Ensures the signal is significant relative to noise.
    df['vol_force'] = df['micro_force'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle potential infs from division or NaNs from diffs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Force Convergence (MFC)".

    Logic:
    - Calculate Expected Value (EV) for both sides.
    - Identify the 'Microstructure Force' from the engineered features.
    - Alignment Check: If the model predicts 'Up' but the force is negative (divergence), 
      drastically reduce the bet size. If they align, increase conviction.
    - Time-Gating: Reduce exposure as the market window reaches its final seconds.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract the engineered force signal
    vol_force = features_row.get('vol_force', 0.0)

    # 1. Determine Alignment
    # Alignment is true if the predicted direction matches the microstructure force direction.
    is_aligned = (side == 'Up' and vol_force > 0) or (side == 'Down' and vol_force < 0)

    # 2. Decision Threshold and Scaling
    # We require a minimum 3% edge to overcome noise.
    if best_ev > 0.03:
        # If aligned, we are aggressive (1.3x). If diverging, we are extremely cautious (0.3x).
        multiplier = 1.3 if is_aligned else 0.3
        
        # 3. Time-Gating
        # In the final 30 seconds, volatility spikes and spreads widen; scale down.
        if seconds_remaining < 30:
            multiplier *= 0.5

        # Scale bet size by the magnitude of the edge relative to a 5% target
        bet_fraction = multiplier * (best_ev / 0.05)
        return side, min(1.0, bet_fraction)

    return None, 0.0