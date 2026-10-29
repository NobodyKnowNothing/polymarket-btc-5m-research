# Var 145 | BT: 0.222 | Live PnL: $-27.39 | Trades: 60
# Desc: Implements "Cross-Venue Stability & Flow-Intent Alignment (CV-SFIA)," which filters for high-conviction trades by requiring low cross-exchange price dispersion (stability) and high alignment between a
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, -1.0, -1.0, -1.0, 1.0850685834884644, -1.0, -1.0, 0.04840129613876343, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.890848, -1.0, -1.0, -0.9502360224723816, -1.0, -1.0, -1.0, 1.3396370000000002, -1.0, 1.092555, 2.651597, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -0.8047058582305908, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8578130602836609, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, 0.10316799999999993, 0.06987699999999997, 2.6440839767456055, -1.0, -1.0, -1.0], "live_pnl": -27.393260955810547, "live_wins": 13, "live_total": 60, "age": 125, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 0.22198571343877826, "backtest_acc": 0.580488750969744, "backtest_n": 2977}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange stability and the coherence between flow and intent.
    """
    df = df.copy()

    # 1. Normalized Price Dispersion: How much do exchanges disagree on price?
    # Lower is better (more stability).
    df['normalized_price_spread'] = df['price_spread'] / (df['binance_price'] + 1e-6)

    # 2. Flow-Intent Coherence: Do aggressive trades (Delta) align with book imbalance (OBI)?
    # Positive = high alignment (buying pressure + bid imbalance), Negative = high alignment (selling pressure + ask imbalance).
    df['flow_intent_coherence'] = df['mean_delta'] * df['mean_obi']

    # 3. Momentum-Flow Alignment: Is the recent price trend supported by aggressive flow?
    df['momentum_flow_alignment'] = df['mean_momentum'] * df['mean_delta']

    # 4. Volatility-Adjusted Spread: Is the spread wide relative to current volatility?
    df['vol_adjusted_spread'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Venue Stability & Flow-Intent Alignment (CV-SFIA)".

    Logic:
    - Calculate raw EV for both sides.
    - Stability Factor: Penalizes trades when cross-exchange price dispersion is high.
      High spread = high uncertainty/noise.
    - Alignment Factor: A binary gate that heavily penalizes trades where 
      aggressive flow (Delta) and passive intent (OBI) are in conflict.
    - Effective EV: Scaled by both stability and alignment.
    - Bet Sizing: Scales linearly with stability; we only bet big when the market is synchronized.
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
    norm_spread = features_row.get('normalized_price_spread', 0.0)
    flow_intent = features_row.get('flow_intent_coherence', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)

    # 1. Stability Factor (0.0 to 1.0)
    # Uses an exponential decay on the normalized spread to heavily penalize divergence.
    stability_factor = math.exp(-norm_spread * 500)

    # 2. Alignment Factor (0.2 to 1.0)
    # If Delta and OBI have the same sign, the move is "coherent".
    # If they conflict, the move is likely "absorption" or "noise".
    if (mean_delta > 0 and mean_obi > 0) or (mean_delta < 0 and mean_obi < 0):
        alignment_factor = 1.0
    elif (mean_delta == 0) or (mean_obi == 0):
        alignment_factor = 0.5
    else:
        alignment_factor = 0.2

    # 3. Calculate Effective EV
    effective_ev = best_ev * stability_factor * alignment_factor

    # 4. Execution Logic
    # We require a higher bar for effective edge.
    threshold = 0.03
    if seconds_remaining < 45:
        threshold = 0.045 # Increase hurdle as time runs out to avoid last-second noise

    if effective_ev > threshold:
        # Bet fraction scales with effective EV but is capped and dampened by stability.
        # A "perfect" trade (EV=0.04, stability=1.0, alignment=1.0) gives 1.0 bet.
        bet_fraction = (effective_ev / 0.04) * stability_factor
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0