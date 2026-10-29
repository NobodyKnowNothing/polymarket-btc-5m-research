# Var 144 | BT: -0.984 | Live PnL: $-23.19 | Trades: 61
# Desc: Implements "Dynamic Liquidity-Momentum Gating (DLMG)," which uses a non-linear, adaptive EV threshold that scales inversely with microstructure momentum and liquidity, while increasing the hurdle as t
# Stats: {"live_returns": [-0.7967209815979004, 1.7294070000000001, -1.0, 0.04551428556442261, -0.9983009099960327, -1.0, -1.0, 0.06848162412643433, 0.08073200000000003, -1.0, -0.9105607867240906, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, 1.233902, -1.0, 0.04551428556442261, -1.0, 0.15118699999999996, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -0.9983009099960327, -1.0, -1.0, 0.04551428556442261, 2.950496, 0.16389999999999993, -1.0, -1.0, -1.0, 1.890848, -1.0, -1.0, -0.9983009099960327, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, 1.28551, -1.0, 2.073133, -1.0, -0.7731261849403381, 2.651597, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -1.0], "live_pnl": -23.191076278686523, "live_wins": 18, "live_total": 61, "age": 101, "rounds_underwater": 59, "abstain_rounds": 0, "backtest_score": -0.9839044665063937, "backtest_acc": 0.5425904317386231, "backtest_n": 4253}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the interaction between momentum, liquidity, and friction.
    """
    df = df.copy()

    # 1. Microstructure Momentum: Alignment of flow (Delta) and intent (OBI).
    # High positive = strong buying pressure; High negative = strong selling pressure.
    df['microstructure_momentum'] = df['mean_delta'] * df['mean_obi']

    # 2. Market Friction: Combined measure of price and book dispersion.
    # Higher values indicate a noisier, less liquid environment.
    df['market_friction'] = df['price_spread'] + df['obi_spread']

    # 3. Liquidity Score: Inverse of friction, normalized to [0, 1].
    df['liquidity_score'] = 1.0 / (1.0 + df['market_friction'])

    # 4. Delta Velocity: The rate of change of aggressive flow within the window.
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 5. Momentum Acceleration: The rate of change of the momentum index.
    df['momentum_acceleration'] = df.groupby('market_slug')['microstructure_momentum'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return robust hyperparameters with moderate regularization."""
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
    Decide whether to trade using "Dynamic Liquidity-Momentum Gating (DLMG)".

    Logic:
    1. Calculate Base EV: Standard (P - Cost) for both sides.
    2. Dynamic Hurdle: Instead of a static 0.03 threshold, the required EV is:
       Threshold = 0.03 / ((1 + |Momentum|) * (1 + Liquidity))
       - High momentum or high liquidity lowers the hurdle (more conviction).
       - Low momentum or low liquidity raises the hurdle (more uncertainty).
    3. Temporal Penalty: As 'seconds_remaining' approaches 0, the hurdle increases 
       to protect against the high-entropy "coin flip" nature of the final seconds.
    4. Bet Sizing: Scale bet size by the ratio of realized EV to a target 5% edge,
       and decay the size significantly in the final 60 seconds.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    momentum = features_row.get('microstructure_momentum', 0.0)
    liquidity = features_row.get('liquidity_score', 0.0)
    
    # 1. Calculate Dynamic Threshold
    # We want the threshold to be lower when momentum and liquidity are high.
    base_threshold = 0.03
    momentum_factor = 1.0 + abs(momentum)
    liquidity_factor = 1.0 + liquidity
    
    dynamic_threshold = base_threshold / (momentum_factor * liquidity_factor)
    
    # 2. Temporal Hurdle Scaling
    # As time runs out (300s -> 0s), the threshold increases (up to 2x).
    time_multiplier = 1.0 + (300.0 - seconds_remaining) / 300.0
    final_threshold = dynamic_threshold * time_multiplier

    # 3. Execution Logic
    side = None
    best_ev = 0.0
    
    if ev_up > ev_down:
        if ev_up > final_threshold:
            side = 'Up'
            best_ev = ev_up
    else:
        if ev_down > final_threshold:
            side = 'Down'
            best_ev = ev_down

    if side:
        # 4. Dynamic Bet Sizing
        # Scale by edge magnitude (target 5% edge for full bet)
        edge_scale = best_ev / 0.05
        
        # Scale down significantly in the final minute to avoid noise
        time_decay = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        bet_fraction = edge_scale * time_decay
        return side, min(1.0, max(0.01, bet_fraction))

    return None, 0.0