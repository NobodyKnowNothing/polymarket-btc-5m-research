# Var 123 | BT: -0.950 | Live PnL: $-1.42 | Trades: 70
# Desc: Implements "Microstructure Coiling & Polymarket Value Gap (CPVG)," which identifies high-pressure buildup (high acceleration of the OBI-Delta product during low-volatility regimes) to anticipate explo
# Stats: {"live_returns": [-1.0, 1.233902, -1.0, 0.08067667484283447, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 0.08073200000000003, 0.03982502222061157, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -0.8469976782798767, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 1.28551, 2.795066, 3.727685, 3.1194439999999997, 1.3396370000000002, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.4199409484863281, "live_wins": 22, "live_total": 70, "age": 115, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.9496950066795662, "backtest_acc": 0.5601006922592826, "backtest_n": 3483}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the acceleration of order book pressure relative to price volatility.
    """
    df = df.copy()

    # 1. Pressure Index: The product of book imbalance and aggressive flow.
    # High positive = aggressive buying into a heavy bid book.
    # High negative = aggressive selling into a heavy ask book.
    df['pressure_index'] = df['mean_obi'] * df['mean_delta']

    # 2. Pressure Acceleration: The rate of change of the pressure index.
    # This captures the "coiling" effect where pressure builds up rapidly.
    df['pressure_accel'] = df.groupby('market_slug')['pressure_index'].diff().fillna(0)

    # 3. Flow Efficiency: How much delta is actually moving the price per unit of volatility.
    # High efficiency = "Vacuum" (price moves easily with little flow).
    # Low efficiency = "Absorption" (heavy flow, little price movement).
    df['price_efficiency'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Cross-Exchange Tension: Divergence in OBI across the three major venues.
    df['obi_tension'] = df['obi_spread'] * df['mean_volatility']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Coiling & Polymarket Value Gap (CPVG)".

    Logic:
    - Coiling Regime: High pressure acceleration + Low volatility. 
      The market is "coiling" for a breakout. Lower EV threshold, higher bet size.
    - Breakout Regime: High pressure acceleration + High efficiency. 
      The move is already happening. Higher EV threshold, lower bet size (avoid chasing).
    - Absorption Regime: Low efficiency (High Delta, Low Price move).
      The market is hitting a wall. High EV threshold required.
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
    p_accel = features_row.get('pressure_accel', 0.0)
    efficiency = features_row.get('price_efficiency', 0.0)
    vol = features_row.get('mean_volatility', 0.01)
    
    # Normalize acceleration by volatility to find "Coiling Intensity"
    coiling_intensity = abs(p_accel) / (vol + 1e-6)

    # 1. Identify Regime
    # Coiling: Pressure is building but price hasn't reacted (Low efficiency/vol)
    if coiling_intensity > 1.5 and abs(efficiency) < 0.5:
        req_ev = 0.022  # Aggressive entry for the "coiled spring"
        multiplier = 1.3
    # Breakout: Pressure is building and price is already moving (High efficiency)
    elif coiling_intensity > 1.5 and abs(efficiency) >= 0.5:
        req_ev = 0.045  # Conservative entry to avoid chasing the peak
        multiplier = 0.7
    # Standard: Normal market conditions
    else:
        req_ev = 0.035
        multiplier = 1.0

    # 2. Time-Gating: In the final 30 seconds, volatility spikes and signals decay.
    if seconds_remaining < 30:
        req_ev *= 1.5
        multiplier *= 0.5

    # 3. Execution
    if best_ev > req_ev:
        # Scale bet size based on the magnitude of the edge relative to a 4% baseline
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0