# Var 182 | BT: -0.105 | Live PnL: $-0.97 | Trades: 74
# Desc: Convergence-Validated Momentum (CVM) filters momentum signals by requiring price-spread contraction, ensuring trades are only taken when multi-exchange consensus is tightening and the move is not a lo
# Stats: {"live_returns": [-1.0, -0.07807708531618118, -1.0, -1.0, -1.0, -0.16567625105381012, -0.2676158547401428, -1.0, -1.0, -1.0, -1.0, -0.1370212286710739, -1.0, -0.043259840458631516, -1.0, -0.9865528345108032, -0.3878064453601837, 0.04989507794380188, -0.011661381460726261, -1.0, 0.0012662999999999997, -0.055565834045410156, -0.3986702859401703, -1.0, 3.1194439999999997, -1.0, 0.27863848209381104, 0.15118699999999996, -0.01, 12.389175, -0.03364858031272888, -1.0, -1.0, -0.09521638602018356, -0.07582513988018036, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 1.7294070000000001, -1.0, 0.11693847924470901, -0.018271757289767265, -1.0, 0.7774268388748169, -1.0, 1.9791409999999998, 0.5874288082122803, 0.3324428200721741, 0.14405465126037598, -0.1578567773103714, -1.0, 1.5854279999999998, -1.0, 1.8077519999999998, 1.3964720000000002, 2.950496, 0.013396370000000003, 1.7294070000000001, -1.0, -0.11437832564115524, -0.069938525557518, 0.11476700000000006, -0.016342030838131905, -1.0, -0.04414839297533035, -0.01919892244040966, -0.6418569087982178, -1.0, 0.016020774841308594, -1.0], "live_pnl": -0.9729752540588379, "live_wins": 22, "live_total": 74, "age": 116, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": -0.10514699998387977, "backtest_acc": 0.5888426009780837, "backtest_n": 4359}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    velocity of convergence/divergence between exchanges.
    """
    df = df.copy()

    # 1. Spread Velocity: Is the price dispersion across exchanges shrinking (convergence) 
    # or growing (divergence)? Convergence (negative velocity) implies consensus.
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 2. Order Book Velocity: The rate of change in the aggregate OBI.
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 3. Delta Velocity: The acceleration of aggressive trade flow.
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 4. Funding Velocity: Is funding pressure accelerating?
    df['funding_velocity'] = df.groupby('market_slug')['mean_funding'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to handle the high-noise velocity features."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using the Convergence-Validated Momentum (CVM) logic.

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Regime Check: Determine if we are in a 'Consensus' (spread_velocity <= 0) 
       or 'Divergence' (spread_velocity > 0) regime.
    3. Adaptive Hurdle: Require a much higher EV in Divergence regimes to 
       protect against localized exchange spikes/noise.
    4. Dynamic Sizing: Scale bet size inversely with spread velocity to 
       maximize Sharpe by avoiding 'chaotic' widening spreads.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract velocity feature
    sv = features_row.get('spread_velocity', 0.0)
    
    side = None
    best_ev = 0.0

    # 1. Regime-Adaptive Hurdle
    # In divergence (sv > 0), we need more edge to offset the risk of a fake-out.
    if sv > 0.001:
        hurdle = 0.045
    else:
        hurdle = 0.025

    # 2. Directional Selection
    if ev_up > ev_down and ev_up > hurdle:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up and ev_down > hurdle:
        side = 'Down'
        best_ev = ev_down

    # 3. Execution and Sizing
    if side:
        # Base scale on edge magnitude (target 5% EV for full bet)
        edge_scale = best_ev / 0.05

        # Convergence Scale: If spread is widening (sv > 0), aggressively penalize bet size.
        # If spread is contracting (sv <= 0), maintain full confidence.
        convergence_penalty = 1.0 / (1.0 + max(0, sv * 20.0))
        
        # Time factor: avoid the very end of the window where volatility spikes
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)

        bet_fraction = edge_scale * convergence_penalty * time_factor

        return side, min(1.0, max(0.01, bet_fraction))

    return None, 0.0