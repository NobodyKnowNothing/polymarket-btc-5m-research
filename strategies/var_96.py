# Var 96 | BT: -0.025 | Live PnL: $-6.58 | Trades: 61
# Desc: Implements "Liquidation-Driven Momentum (LDM)," which identifies high-conviction trades where aggressive delta flow forces the liquidation of crowded perp positions (indicated by funding rates).
# Stats: {"live_returns": [-0.4, 0.41978080000000006, -0.4, -0.4, 0.9122292, 0.060474799999999995, 0.019542400000000015, 0.055506399999999956, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.6341712, 0.06555999999999995, -0.4, 0.4550295999999999, -0.4, -0.4, 0.045906800000000025, -0.4, -0.4, -0.4, -0.4, 0.09182599999999996, 1.1846420000000002, -1.0, 3.1194439999999997, -0.4, 0.036730399999999996, -0.4, -0.4, -0.4, -0.4, -1.0, 0.9581640000000001, 0.514204, -0.4, -0.4, 0.58249, -0.4, -1.0, -0.4, -0.4, -0.4, -0.4, 0.7916564, -0.4, -0.4, -0.4, 0.038675000000000015, -0.4, -1.0, -0.4, 0.8292532, -0.4, 1.40225, -0.4, -0.4], "live_pnl": -6.582463800000003, "live_wins": 20, "live_total": 61, "age": 107, "rounds_underwater": 53, "abstain_rounds": 0, "backtest_score": -0.024564468693776015, "backtest_acc": 0.5564725300636643, "backtest_n": 3183}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding 
    interaction terms that capture the synergy between aggressive flow 
    and perp funding sentiment to detect liquidation cascades.
    """
    df = df.copy()

    # 1. Funding-Delta Synchronicity:
    # Captures the alignment (or divergence) between aggressive flow and perp positioning.
    # A large negative value suggests aggressive selling into a long-heavy market (long liquidation).
    df['funding_delta_sync'] = df['mean_funding'] * df['mean_delta']

    # 2. Delta-Volatility Efficiency:
    # Measures how much aggressive flow is being applied relative to the current volatility.
    # High ratio = "clean" aggressive moves; Low ratio = "noisy" or absorbed moves.
    df['delta_vol_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. OBI-Delta Congruence:
    # Measures if the order book pressure (OBI) and aggressive flow (Delta) are moving together.
    df['obi_delta_sync'] = df['mean_obi'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters tuned for capturing non-linear interaction effects."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidation-Driven Momentum (LDM)".

    Logic:
    - Calculate raw EV for both sides.
    - Identify "Liquidation Regimes":
        - Long Liquidation: Positive funding (longs paying) + Negative Delta (aggressive selling).
        - Short Liquidation: Negative funding (shorts paying) + Positive Delta (aggressive buying).
    - If in a liquidation regime, significantly boost the bet size if the model's 
      EV aligns with the liquidation direction.
    - Use a time-gate: As the window closes (< 30s), revert to conservative betting 
      to avoid "pinning" noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract microstructure components from engineered features
    funding = features_row.get('mean_funding', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    vol = features_row.get('mean_volatility', 1e-6)

    # 1. Define Liquidation Regimes
    # We use a small threshold for funding to ensure we are in a truly crowded position
    is_long_liquidation = (funding > 0.00005) and (delta < 0)
    is_short_liquidation = (funding < -0.00005) and (delta > 0)

    # 2. Time-Gating: Reduce conviction as the market approaches resolution
    # Most microstructure signals decay or become noise in the final 30 seconds.
    is_late_stage = seconds_remaining < 30

    # 3. Decision Logic
    
    # CASE A: Liquidation Momentum (High Conviction)
    if not is_late_stage:
        # Long Liquidation -> Bet Down
        if is_long_liquidation and ev_down > 0.02:
            # Scale confidence by how much delta exceeds volatility
            conf = min(1.0, abs(delta) / (vol + 1e-6))
            return 'Down', 0.5 + (0.5 * conf)

        # Short Liquidation -> Bet Up
        if is_short_liquidation and ev_up > 0.02:
            conf = min(1.0, abs(delta) / (vol + 1e-6))
            return 'Up', 0.5 + (0.5 * conf)

    # CASE B: Standard Momentum (Medium Conviction)
    # If no liquidation is detected, use higher EV thresholds and lower bet sizes.
    if not is_late_stage:
        if ev_up > 0.04:
            return 'Up', 0.4
        if ev_down > 0.04:
            return 'Down', 0.4

    # CASE C: Late Stage / Low Edge (Abstain)
    return None, 0.0