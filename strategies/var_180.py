# Var 180 | BT: 0.182 | Live PnL: $-7.31 | Trades: 29
# Desc: Implements "Microstructure Stability & Convergence (MSC)," which filters trades by requiring high directional coherence between OBI, Delta, and Price, while simultaneously ensuring that exchange price
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 0.05925399999999992, -1.0, 0.04601699113845825, 0.030669450759887695, 1.3964720000000002, 0.11476700000000006, 0.028705999999999898, -1.0, -1.0, 0.07966405153274536, 0.11476700000000006, -1.0, -1.0, -1.0, 1.6554160000000002, 1.4562249999999999, 0.1769179999999999, -1.0, -1.0, -1.0, 1.890848, 0.15118699999999996, 0.48840975761413574, -1.0, -1.0, -1.0], "live_pnl": -7.3106794357299805, "live_wins": 14, "live_total": 29, "age": 581, "rounds_underwater": 29, "abstain_rounds": 59, "backtest_score": 0.18184829488093088, "backtest_acc": 0.5385034507809662, "backtest_n": 216}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the velocity of microstructure components and their convergence.
    """
    df = df.copy()

    # 1. Temporal Velocity Features (Grouped by market to prevent leakage)
    # We look at how fast the intent (OBI) and action (Delta) are changing.
    group = df.groupby('market_slug')
    
    df['price_change'] = group['binance_price'].diff().fillna(0)
    df['obi_velocity'] = group['mean_obi'].diff().fillna(0)
    df['delta_velocity'] = group['mean_delta'].diff().fillna(0)
    df['spread_velocity'] = group['price_spread'].diff().fillna(0)

    # 2. Coherence Index: Alignment of Delta, OBI velocity, and Price change.
    # Range [-3, 3]. 3 = All three moving up; -3 = All three moving down.
    df['coherence_index'] = (
        np.sign(df['mean_delta']) + 
        np.sign(df['obi_velocity']) + 
        np.sign(df['price_change'])
    )

    # 3. Convergence Signal: Positive if exchange spreads are narrowing.
    # Narrowing spreads indicate market agreement/stability.
    df['convergence_signal'] = -df['spread_velocity']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
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
        'learning_rate': 0.009,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Stability & Convergence (MSC)".

    Logic:
    1. Identify Model Direction: Which side has the highest EV?
    2. Check Coherence: Does the model's direction align with the 
       microstructure (OBI, Delta, Price)? We require a coherence score 
       of at least 2 (e.g., 2 out of 3 signals align).
    3. Check Convergence: Are exchange spreads narrowing? We only trade 
       when convergence > 0 to avoid high-volatility arbitrage noise.
    4. Dynamic Sizing: Scale by EV, Coherence magnitude, and Convergence.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine the model's preferred direction
    if ev_up > ev_down:
        target_side = 'Up'
        best_ev = ev_up
        target_dir = 1
    elif ev_down > ev_up:
        target_side = 'Down'
        best_ev = ev_down
        target_dir = -1
    else:
        return None, 0.0

    # Extract engineered features
    coherence = features_row.get('coherence_index', 0.0)
    convergence = features_row.get('convergence_signal', 0.0)

    # 1. The "Stability Gate"
    # We only trade if:
    # a) The model's direction is supported by at least 2/3 microstructure signals (coherence * target_dir >= 2)
    # b) The exchange spreads are narrowing (convergence > 0)
    # c) There is a minimum threshold of edge (EV > 0.025)
    
    is_coherent = (coherence * target_dir) >= 2
    is_converging = convergence > 0
    has_edge = best_ev > 0.025

    if is_coherent and is_converging and has_edge:
        # 2. Execution & Sizing
        # Scale bet by:
        # - Magnitude of edge (target 4% EV for full bet)
        # - Coherence strength (3 is better than 2)
        # - Convergence strength (more convergence = more confidence)
        # - Time decay (reduce size in the final 45 seconds)
        
        time_factor = 1.0 if seconds_remaining > 45 else (seconds_remaining / 45.0)
        coherence_multiplier = abs(coherence) / 3.0
        # Use a log-like scaling for convergence to prevent extreme bets
        convergence_multiplier = math.log1p(max(0, convergence)) / 0.5 
        
        # Cap convergence multiplier to avoid over-leveraging on tiny spread changes
        convergence_multiplier = min(1.5, max(0.5, convergence_multiplier))

        bet_fraction = (best_ev / 0.04) * coherence_multiplier * convergence_multiplier * time_factor
        
        return target_side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0