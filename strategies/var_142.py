# Var 142 | BT: -0.370 | Live PnL: $-25.90 | Trades: 61
# Desc: Implements "Regime-Adaptive Efficiency Gating (RAEG-3)," which distinguishes between "Absorption" (high delta, low price movement) and "Vacuum/Trend" (high efficiency) regimes, applying much stricter
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 0.9115316271781921, -1.0, -1.0, -1.0, 0.10316799999999993, 1.8077519999999998, 0.9115316271781921, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, 0.10029178857803345, -1.0, -1.0, -1.0, -0.9531726241111755, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, 0.9115316271781921, -1.0, -0.9957875609397888, -1.0, -1.0, 2.950496, -1.0, -1.0, -1.0, 0.9607979655265808, 1.890848, -1.0, -1.0, -0.9957875609397888, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, -1.0, 2.073133, -1.0, 2.651597, -0.844602644443512, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -25.899965286254883, "live_wins": 15, "live_total": 61, "age": 114, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.36998504633179863, "backtest_acc": 0.5387337790011797, "backtest_n": 3744}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of order flow relative to price displacement.
    """
    # Create a copy to avoid modifying the original df
    X = df.copy()

    # 1. Price-Flow Efficiency: Measures how much price moves per unit of aggressive delta.
    # Low efficiency = Absorption (heavy fighting in the book, price stuck).
    # High efficiency = Vacuum (price moving easily on low liquidity).
    # We use abs() for the feature to capture the magnitude of the relationship.
    X['efficiency'] = (X['price_vs_strike_pct'].abs() / (X['sum_delta'].abs() + 1e-6))

    # 2. Flow Intensity: The raw power of the current microstructure state.
    # Combines the net aggressive flow with the existing book imbalance.
    X['flow_intensity'] = X['sum_delta'].abs() * X['mean_obi'].abs()

    # 3. Volatility-Adjusted Flow: Normalizes flow intensity by the current volatility regime.
    # High value = Significant flow in a quiet market.
    X['vol_pressure'] = X['flow_intensity'] / (X['mean_volatility'] + 1e-6)

    # Drop metadata and target as per instructions
    # Note: We keep 'seconds_remaining', 'pm_cost_up', and 'pm_cost_down' as they are useful for the filter
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
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Adaptive Efficiency Gating (RAEG-3)".

    Logic:
    - Calculate EV for both sides.
    - Identify the Microstructure Regime via 'efficiency':
        - Absorption (eff < 0.5): High delta, low price move. High risk of reversal.
        - Normal (0.5 <= eff <= 2.0): Standard market conditions.
        - Vacuum/Trend (eff > 2.0): High price move, low delta. High momentum/low liquidity.
    - Apply Regime-Specific Hurdles:
        - Absorption requires a much higher EV (0.05) to compensate for uncertainty.
        - Vacuum/Trend allows for a lower EV (0.02) due to high momentum conviction.
    - Dynamic Bet Sizing:
        - Scale bet size based on efficiency and the magnitude of the edge.
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
    efficiency = features_row.get('efficiency', 1.0)
    
    # 1. Regime-Based Hurdle Selection
    if efficiency < 0.5:
        # Absorption Regime: High uncertainty, require high edge
        hurdle = 0.05
        multiplier = 0.6  # Reduce bet size in absorption
    elif efficiency > 2.0:
        # Vacuum/Trend Regime: High momentum, lower hurdle
        hurdle = 0.02
        multiplier = 1.2  # Amplify bet size in vacuum
    else:
        # Normal Regime
        hurdle = 0.03
        multiplier = 1.0

    # 2. Execution Decision
    if best_ev > hurdle:
        # 3. Dynamic Bet Sizing
        # Target a 4% edge for a standard bet, scaled by regime multiplier
        bet_fraction = (best_ev / 0.04) * multiplier
        
        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0