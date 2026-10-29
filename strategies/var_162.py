# Var 162 | BT: 0.501 | Live PnL: $-9.57 | Trades: 60
# Desc: Implements "Order-Flow Divergence Scaling (OFDS)," which bifurcates strategy into "Efficiency-Trend" (following aligned flow) and "Exhaustion-Reversal" (betting against high-delta/low-alignment moves)
# Stats: {"live_returns": [0.2271265983581543, 0.5849659442901611, 0.6341948509216309, -0.288149356842041, -0.8711177110671997, -0.2012215256690979, -0.2124186009168625, -0.41567325592041016, 0.08690092712640762, -0.42570245265960693, -0.14745011925697327, -0.9042973518371582, -0.3708241879940033, -0.22820502519607544, 2.3221371173858643, -0.8790121078491211, -0.16562528908252716, -0.666219174861908, -0.28998953104019165, 1.5609897375106812, -1.0, -0.4684908986091614, 0.0069578588008880615, -0.35996702313423157, -0.17015883326530457, -0.18324686586856842, -0.31750625371932983, 0.01022215187549591, -0.4884604215621948, -0.13437779247760773, -0.15308888256549835, -1.0, 0.03902032971382141, 0.0045912999999999995, -0.9409117698669434, -0.32232120633125305, -0.16522642970085144, -0.7757362127304077, -1.0, 1.4562249999999999, -0.10857000946998596, 0.015338301658630371, -0.050774283707141876, -0.48203086853027344, -0.8708944320678711, -0.23060652613639832, -0.4348016083240509, -0.5320233106613159, 0.0024263933300971985, -0.959843099117279, -0.6138684749603271, -0.16943751275539398, 0.08680660277605057, -0.17364314198493958, 0.05462775, 0.906150221824646, 0.28554868698120117, -0.05, 0.07554744929075241, 0.29324543476104736], "live_pnl": -9.568870544433594, "live_wins": 19, "live_total": 60, "age": 463, "rounds_underwater": 54, "abstain_rounds": 0, "backtest_score": 0.5010444091771177, "backtest_acc": 0.557065723328989, "backtest_n": 817}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between aggressive flow (Delta) and passive intent (OBI).
    """
    df = df.copy()

    # 1. Flow-Intent Alignment: Product of OBI and Delta.
    # Positive = Trend (Flow and Intent agree). Negative = Tension (Flow and Intent conflict).
    df['delta_obi_alignment'] = df['mean_delta'] * df['mean_obi']

    # 2. Flow-Intent Efficiency: How much delta is produced per unit of OBI.
    # High efficiency = price moving easily (low resistance). 
    # Low/Negative efficiency = price fighting a heavy book (high resistance).
    df['delta_obi_efficiency'] = df['mean_delta'] / (df['mean_obi'].abs() + 0.01)

    # 3. Volatility-Adjusted Delta: Normalizes aggressive flow by the current regime.
    df['vol_delta_interaction'] = df['mean_volatility'] * df['mean_delta'].abs()

    # 4. Funding-Flow Divergence: Captures if aggressive flow is moving against funding.
    df['funding_flow_divergence'] = df['mean_funding'] * df['mean_delta']

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
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 35.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Order-Flow Divergence Scaling (OFDS)".

    Logic:
    1. Efficiency-Trend Mode: 
       If Delta and OBI are aligned (positive alignment) and efficient (high ratio),
       we follow the momentum.
    2. Exhaustion-Reversal Mode:
       If Delta is high but Alignment is negative (the 'Tug-of-War'), 
       the aggressive side is likely hitting a wall. We bet the reversal.
    3. Dynamic Sizing:
       Scale bet size by EV, market uncertainty (spread), and time decay.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    alignment = features_row.get('delta_obi_alignment', 0.0)
    efficiency = features_row.get('delta_obi_efficiency', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    price_spread = features_row.get('price_spread', 0.0)
    
    side = None
    best_ev = 0.0

    # Thresholds
    TREND_ALIGNMENT_MIN = 0.02
    TREND_EFFICIENCY_MIN = 0.5
    EXHAUSTION_DELTA_MIN = 0.4
    EXHAUSTION_ALIGNMENT_MAX = -0.02

    # --- MODE 1: EFFICIENCY-TREND (Follow the Flow) ---
    if alignment > TREND_ALIGNMENT_MIN and efficiency > TREND_EFFICIENCY_MIN:
        if mean_delta > 0 and ev_up > 0.03:
            side = 'Up'
            best_ev = ev_up
        elif mean_delta < 0 and ev_down > 0.03:
            side = 'Down'
            best_ev = ev_down

    # --- MODE 2: EXHAUSTION-REVERSAL (Against the Flow) ---
    elif abs(mean_delta) > EXHAUSTION_DELTA_MIN and alignment < EXHAUSTION_ALIGNMENT_MAX:
        # If buyers are aggressive but OBI is negative, bet Down
        if mean_delta > 0 and ev_down > 0.04:
            side = 'Down'
            best_ev = ev_down
        # If sellers are aggressive but OBI is positive, bet Up
        elif mean_delta < 0 and ev_up > 0.04:
            side = 'Up'
            best_ev = ev_up

    # --- EXECUTION ---
    if side:
        # 1. Magnitude of edge (Target 5% EV for full bet)
        edge_scale = best_ev / 0.05
        
        # 2. Uncertainty Penalty (Inverse of price spread)
        uncertainty_scale = 1.0 / (1.0 + price_spread)
        
        # 3. Time Decay (Reduce size in the final 2 minutes to avoid volatility spikes)
        time_scale = 1.0 if seconds_remaining > 120 else (seconds_remaining / 120.0)
        
        bet_fraction = edge_scale * uncertainty_scale * time_scale
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0