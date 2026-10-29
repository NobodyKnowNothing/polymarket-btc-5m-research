# Var 170 | BT: 0.057 | Live PnL: $-0.04 | Trades: 60
# Desc: Regime-Adaptive Divergence Scaling (RADS): Uses the alignment between aggressive flow and price movement to switch between trend-following and reversal-hunting regimes, with a time-decaying hurdle to 
# Stats: {"live_returns": [-1.0, 1.3964720000000002, -1.0, -1.0, -1.0, 0.12663000000000002, 1.28551, -1.0, 0.12663000000000002, 0.15118699999999996, -0.9159632325172424, -1.0, -1.0, -1.0, 1.890848, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.4562249999999999, 4.245841, -1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, 1.233902, 2.280573, -1.0, -1.0, 1.0898240000000001, -1.0, -1.0, -1.0, 0.04885600000000001, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, 9.427877, 1.5191249999999998, -1.0, -1.0], "live_pnl": -0.043570518493652344, "live_wins": 20, "live_total": 60, "age": 96, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.05653244376215582, "backtest_acc": 0.5853387634209553, "backtest_n": 4017}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between aggressive flow (Delta) and price displacement.
    """
    df = df.copy()

    # 1. Delta-Price Coherence: Do aggressive trades align with price movement?
    # Positive = Trend (Delta and Price move together), Negative = Divergence (Absorption/Reversal)
    df['delta_price_sync'] = np.sign(df['sum_delta']) * np.sign(df['price_vs_strike_pct'])

    # 2. Microstructure Force: The magnitude of order flow pressure relative to volatility.
    # This helps the model distinguish between high-conviction moves and noisy fluctuations.
    df['microstructure_force'] = (df['sum_delta'] * df['mean_obi']) / (df['mean_volatility'] + 1e-6)

    # 3. Funding-Delta Interaction: Capturing potential squeeze/exhaustion dynamics.
    df['funding_delta_interaction'] = df['mean_funding'] * df['sum_delta']

    # Drop metadata and target
    # We keep pm_cost_up/down as they are useful for the model to learn market pricing
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Adaptive Divergence Scaling (RADS)".

    Logic:
    1. Identify Microstructure Regime:
       - TREND: Delta and Price movement are aligned (sync > 0). Lower hurdle.
       - DIVERGENCE: Delta and Price movement are fighting (sync < 0). Higher hurdle (reversal play).
       - STAGNANT: No clear direction (sync == 0). Highest hurdle.
    2. Time-Decaying Hurdle:
       - As seconds_remaining decreases, the hurdle increases to avoid late-window noise.
    3. Execution:
       - Scale bet size by the regime multiplier (Trend = 1.0, Divergence = 0.7, Stagnant = 0.5).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    sync = features_row.get('delta_price_sync', 0.0)

    # 1. Regime Identification & Hurdle Setting
    if sync > 0:
        # Trend Regime: Follow the momentum
        base_hurdle = 0.020
        regime_multiplier = 1.0
    elif sync < 0:
        # Divergence Regime: High-risk reversal/absorption play
        base_hurdle = 0.040
        regime_multiplier = 0.7
    else:
        # Stagnant Regime: High uncertainty
        base_hurdle = 0.050
        regime_multiplier = 0.5

    # 2. Time-Decay Penalty
    # As time runs out (seconds_remaining -> 0), the hurdle increases.
    # At 300s, penalty is 1.0. At 0s, penalty is 2.0.
    time_penalty = 1.0 + ((300.0 - seconds_remaining) / 300.0)
    effective_hurdle = base_hurdle * time_penalty

    # 3. Execution
    if best_ev > effective_hurdle:
        # Scale bet size: Target 4% edge for full sizing (1.0), adjusted by regime
        # We clamp the multiplier to ensure we don't over-leverage on tiny edges
        bet_fraction = (best_ev / 0.04) * regime_multiplier
        
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0