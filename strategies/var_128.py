# Var 128 | BT: -0.784 | Live PnL: $-22.01 | Trades: 61
# Desc: Implements "Dispersion-Gated Momentum Alignment (DGMA)," which penalizes expected value when exchange price dispersion is high (indicating market fragmentation/noise) and amplifies it when aggressive
# Stats: {"live_returns": [-1.0, -1.0, 1.9791409999999998, -0.987485945224762, -1.0, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, 2.280573, -1.0, -1.0, 0.03932154178619385, -0.7969606518745422, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -1.0, 0.03334993124008179, -1.0, 0.09182599999999996, -1.0, 2.073133, 2.280573, -1.0, -0.933743953704834, -1.0, -1.0, -1.0, -0.825671374797821, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, 0.12273836135864258, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 0.7651998996734619, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0], "live_pnl": -22.01386833190918, "live_wins": 15, "live_total": 61, "age": 89, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.783583648045638, "backtest_acc": 0.5601388605268531, "backtest_n": 3716}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding 
    dispersion-relative metrics and flow-momentum synergies.
    """
    df = df.copy()

    # 1. Relative Price Dispersion: Normalizes price spread by price level.
    # High values indicate exchange disagreement/fragmentation.
    df['rel_price_std'] = df['price_std'] / df['binance_price'].replace(0, np.nan)

    # 2. Delta-OBI Synergy: Captures if aggressive flow is supported by book imbalance.
    # Positive = aggressive buying into a heavy bid side.
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 3. Momentum-Flow Alignment: Captures if recent price movement is backed by aggressive flow.
    df['momentum_flow_sync'] = df['mean_momentum'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust microstructure signals."""
    return {
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Dispersion-Gated Momentum Alignment (DGMA)".

    Logic:
    - Calculate raw EV for both sides.
    - Dispersion Penalty: Scales EV down if rel_price_std is high (market is fragmented/unstable).
    - Momentum Alignment: Amplifies EV (1.2x) if the side aligns with both momentum and delta.
      Penalizes EV (0.6x) if the side contradicts the momentum/flow trend.
    - Dynamic Thresholding: Increases required edge as the market window closes.
    - Bet Sizing: Scales bet fraction based on the "effective" (penalized/amplified) EV.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 1. Dispersion Penalty
    # Uses rel_price_std to scale down conviction in fragmented markets.
    # A 10bps dispersion (0.001) results in a ~0.66x multiplier.
    rel_std = features_row.get('rel_price_std', 0.0)
    dispersion_penalty = 1.0 / (1.0 + (rel_std * 500))

    # 2. Momentum & Flow Alignment Factor
    # We want the trade direction to be supported by both momentum and delta.
    m_mom = features_row.get('mean_momentum', 0.0)
    m_delta = features_row.get('mean_delta', 0.0)
    
    momentum_factor = 1.0
    if side == 'Up':
        if m_mom > 0 and m_delta > 0:
            momentum_factor = 1.2
        elif m_mom < 0 or m_delta < 0:
            momentum_factor = 0.6
    else: # side == 'Down'
        if m_mom < 0 and m_delta < 0:
            momentum_factor = 1.2
        elif m_mom > 0 or m_delta > 0:
            momentum_factor = 0.6

    # 3. Calculate Effective EV
    effective_ev = best_ev * dispersion_penalty * momentum_factor

    # 4. Dynamic Thresholding & Execution
    # As time runs out, we require higher certainty to avoid "last-second noise".
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05

    if effective_ev > base_threshold:
        # Scale bet fraction: target 5% effective edge for a full bet.
        bet_fraction = effective_ev / 0.05
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0