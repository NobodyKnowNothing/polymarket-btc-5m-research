# Var 139 | BT: -0.365 | Live PnL: $-23.79 | Trades: 61
# Desc: Implements "Microstructure Efficiency & Convergence Gating (MECG)," which targets high-conviction momentum by requiring that aggressive flow (Delta) is moving price efficiently relative to volatility
# Stats: {"live_returns": [-1.0, 3.727685, -1.0, 1.3396370000000002, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, -1.0, -1.0, -1.0, -1.0, 1.1375739999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, 0.13876599999999994, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, 1.1846420000000002, 1.8077519999999998, 1.1846420000000002, -1.0, 1.049452, 1.7294070000000001, 0.04885600000000001, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, 0.15118699999999996, 0.028705999999999898, -1.0], "live_pnl": -23.791871999999998, "live_wins": 18, "live_total": 61, "age": 99, "rounds_underwater": 51, "abstain_rounds": 0, "backtest_score": -0.36540525200595275, "backtest_acc": 0.5734446432120851, "backtest_n": 4153}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of flow and the convergence of exchange spreads.
    """
    df = df.copy()

    # 1. Delta Efficiency: How much delta is required to move price relative to volatility.
    # High value = "Clean" move (high impact, low noise). Low value = "Absorption" (high noise).
    # We clip to prevent extreme outliers from dominating the XGBoost model.
    df['delta_efficiency'] = (df['sum_delta'] / (df['mean_volatility'] + 1e-6)).clip(-5, 5)

    # 2. Spread Convergence: The change in cross-exchange price dispersion.
    # A negative value indicates exchanges are converging on a new price (high conviction).
    df['spread_diff'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 3. OBI-Delta Sync: Do the order book imbalance and aggressive flow agree?
    # Range [-1, 1]. 1 = both bullish, -1 = both bearish, 0 = disagreement.
    df['obi_sync'] = df['mean_obi'] * np.sign(df['sum_delta'])

    # Drop metadata and target as per instructions
    # We keep 'seconds_remaining' as it is a critical market state feature.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 50.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Efficiency & Convergence Gating (MECG)".

    Logic:
    - Calculate raw EV for both sides.
    - Convergence Multiplier: If exchange spreads are tightening (spread_diff <= 0), 
      amplify EV (1.2x). If they are widening (divergence), penalize EV (0.6x).
    - Efficiency Multiplier: Amplify EV based on the magnitude of 'delta_efficiency'.
      This rewards trades where flow is moving price cleanly.
    - Thresholding: Use a dynamic threshold that increases as the market approaches expiry.
    - Bet Sizing: Scale bet fraction by the effective EV, capped at 1.0.
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
    delta_eff = features_row.get('delta_efficiency', 0.0)
    spread_diff = features_row.get('spread_diff', 0.0)
    obi_sync = features_row.get('obi_sync', 0.0)

    # 1. Convergence Factor
    # Tightening spreads = higher conviction in the current price direction.
    convergence_mult = 1.2 if spread_diff <= 0 else 0.6

    # 2. Efficiency Factor
    # High delta_efficiency means the move is "clean" and not being absorbed.
    efficiency_mult = 1.0 + (abs(delta_eff) * 0.15)

    # 3. Calculate Effective EV
    effective_ev = best_ev * convergence_mult * efficiency_mult

    # 4. Dynamic Thresholding
    # As seconds_remaining decreases, we require a higher edge to avoid late-window noise.
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 180:
        base_threshold = 0.04

    # 5. Final Decision
    if effective_ev > base_threshold:
        # Scale bet fraction by effective EV. 
        # We target a 4% effective edge for a full bet.
        # We also add a small boost if OBI and Delta are synchronized.
        sync_boost = 1.0 + (abs(obi_sync) * 0.2)
        bet_fraction = (effective_ev / 0.04) * sync_boost
        
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0