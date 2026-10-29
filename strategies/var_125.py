# Var 125 | BT: -0.152 | Live PnL: $-4.95 | Trades: 61
# Desc: Implements "Microstructure Coherence & Friction Scaling (MCFS)," which distinguishes between high-conviction trends (where price, OBI, and delta align) and high-risk absorption (where aggressive delta
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.028801262378692627, 4.550646, -1.0, -0.6819907426834106, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 1.28551, 3.727685, 3.1194439999999997, 1.3396370000000002, -1.0, 1.9791409999999998, -0.7928885817527771, -1.0, 3.303756, -1.0, 0.05567139387130737, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -0.5183499455451965, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.6554160000000002, -0.992695152759552, -1.0, -1.0, -1.0, 2.073133, -1.0], "live_pnl": -4.949915409088135, "live_wins": 17, "live_total": 61, "age": 98, "rounds_underwater": 11, "abstain_rounds": 0, "backtest_score": -0.15245773946613286, "backtest_acc": 0.5558549222797927, "backtest_n": 3312}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between price movement, order book imbalance, and aggressive flow.
    """
    df = df.copy()

    # 1. Price Velocity: Change in price within the market window
    df['price_diff'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Delta-OBI Sync: Do aggressive trades align with the book imbalance?
    # 1.0 = aligned, -1.0 = opposing, 0.0 = neutral
    df['delta_obi_sync'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 3. Microstructure Coherence: Sum of signs of Delta, OBI, and Price Change
    # 3 or -3: Perfect Trend (All agree)
    # 1 or -1: Mixed/Absorption (One signal disagrees)
    # 0: Neutral/No signal
    df['coherence'] = np.sign(df['mean_delta']) + np.sign(df['mean_obi']) + np.sign(df['price_diff'])

    # 4. Friction Index: How much delta is required to move the price?
    # High friction = Absorption/Walls. Low friction = Liquidity Vacuum.
    df['friction'] = df['mean_delta'].abs() / (df['price_diff'].abs() + 1e-6)

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
    Decide whether to trade using "Microstructure Coherence & Friction Scaling (MCFS)".

    Logic:
    - Strong Trend (Coherence >= 2 or <= -2): High conviction, lower EV hurdle.
    - Absorption (Coherence is mixed AND Delta-OBI sync is negative): High risk, higher EV hurdle.
    - Mixed/Noise: Standard EV hurdle.
    - Friction Penalty: If friction is extremely high (price isn't moving despite delta), 
      drastically reduce bet size to avoid being trapped in absorption.
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
    coherence = features_row.get('coherence', 0.0)
    delta_obi_sync = features_row.get('delta_obi_sync', 0.0)
    friction = features_row.get('friction', 0.0)

    # 1. Determine Regime and set base requirements
    if abs(coherence) >= 2:
        # Strong Trend Regime: All signals (Delta, OBI, Price) are in agreement
        req_ev = 0.020
        multiplier = 1.25
    elif abs(coherence) >= 1:
        # Mixed Regime: One signal is lagging or opposing
        if delta_obi_sync < 0:
            # Absorption Regime: Aggressive flow is fighting the order book
            req_ev = 0.040
            multiplier = 0.50
        else:
            # Mixed Trend: Signals are somewhat aligned but not perfect
            req_ev = 0.030
            multiplier = 0.80
    else:
        # Neutral/No Signal Regime
        return None, 0.0

    # 2. Apply Friction Penalty
    # If friction is very high, the market is absorbing the flow (price is stuck)
    if friction > 100:
        multiplier *= 0.4
        req_ev *= 1.5

    # 3. Time-Gating: Tighten requirements as the window closes
    if seconds_remaining < 60:
        req_ev *= 1.4
        multiplier *= 0.7

    # 4. Final Execution
    if best_ev > req_ev:
        # Scale bet size based on the magnitude of the edge relative to a 4% target
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0