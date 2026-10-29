# Var 203 | BT: 0.112 | Live PnL: $-3.70 | Trades: 60
# Desc: Targets "Lead-Venue Dispersion Breakouts" (LVDB) by identifying moments when cross-exchange price divergence is accelerating in tandem with directional order flow, signaling a high-conviction breakout
# Stats: {"live_returns": [-0.1445744607993989, -0.05, 0.6520181999999999, -0.3, 0.2785036570695445, -0.3, 0.6841718999999999, -0.3, -0.3, -0.3, 0.9358331999999998, -0.3, 0.05462894334968749, -0.3, -0.07384620706678004, -0.22398989759839386, 0.09895705, -0.18462389480016983, -1.0, 0.5672543999999999, -0.25988602500072033, 1.2679283639502172, 0.31483560000000005, -1.0, 0.37017059999999996, -0.3, -0.3, 0.3277665, -0.3, -0.13807150933383563, -0.3, 0.09454240000000001, -0.3, -0.05, 2.651597, -0.5364417739985858, -1.0, -0.3, -0.8685550119995835, 0.08073200000000003, -0.3, -0.3, -0.3, -0.11870995679969061, 0.718623, -0.3, -0.7205382306101381, -0.16585110630003103, -0.3, -0.08783683999978695, -0.3, -0.21093538339912293, -0.3, 0.6520181999999999, -0.6124214023335532, -0.20781386324566922, -0.05, -0.3, 0.2971242586461239, 0.2533344960669346], "live_pnl": -3.7040557942029526, "live_wins": 18, "live_total": 60, "age": 101, "rounds_underwater": 23, "abstain_rounds": 0, "backtest_score": 0.11165733181777979, "backtest_acc": 0.5487676056338028, "backtest_n": 4311}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the acceleration of cross-exchange dispersion as a signal for breakouts.
    """
    df = df.copy()

    # 1. Spread Velocity: Is the disagreement between exchanges growing or shrinking?
    # A growing spread (positive velocity) during a move often signals a lead-venue breakout.
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 2. Flow Intensity: The product of order book intent and aggressive flow.
    # High magnitude indicates strong directional conviction.
    df['flow_intensity'] = df['mean_delta'] * df['mean_obi']

    # 3. Asymmetry Signal: Interaction between dispersion acceleration and flow conviction.
    # This captures the "impulse" of the breakout.
    df['asymmetry_signal'] = df['spread_velocity'] * df['flow_intensity']

    # 4. Volatility-Normalized Dispersion
    df['norm_spread_vel'] = df['spread_velocity'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to capture the transient nature of breakout signals."""
    return {
        'n_estimators': 800,
        'max_depth': 3,
        'learning_rate': 0.015,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Venue Dispersion Breakout (LVDB)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Breakout Detection: 
       - We look for positive 'spread_velocity' (exchanges moving apart).
       - We require 'flow_intensity' to be high (consensus between OBI and Delta).
    3. Dynamic Sizing:
       - Scale bet size by the 'asymmetry_signal' (how much the spread is expanding relative to flow).
       - Apply a time-gate: Breakouts are most reliable in the first 3 minutes; 
         late-window dispersion is often noise or arbitrage.
    """
    # Exit if too little time remains or if in the very last minute (noise zone)
    if seconds_remaining < 60:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Determine the best model-suggested side
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
        flow_dir = 1
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
        flow_dir = -1
    else:
        return None, 0.0

    # Extract engineered features
    spread_vel = features_row.get('spread_velocity', 0.0)
    flow_int = features_row.get('flow_intensity', 0.0)
    asym_sig = features_row.get('asymmetry_signal', 0.0)

    # --- 1. BREAKOUT VALIDATION ---
    # A breakout requires:
    # - Expanding dispersion (spread_vel > 0)
    # - Directional flow matching the model's side (flow_int matches side)
    # - Sufficient model edge (best_ev > 0.025)
    
    is_breakout = (spread_vel > 0) and (flow_int * flow_dir > 0.01)
    
    # If the model sees edge but the microstructure isn't "breaking out", 
    # we increase the hurdle to avoid low-conviction/choppy moves.
    hurdle = 0.03
    if not is_breakout:
        hurdle = 0.05

    if best_ev > hurdle:
        # --- 2. DYNAMIC SIZING ---
        # Base size on the magnitude of the asymmetry signal
        # We use the absolute value of asym_sig to scale the bet.
        # We cap the multiplier to prevent extreme bets.
        size_multiplier = min(1.0, abs(asym_sig) * 5.0)
        
        # If it's a confirmed breakout, we can be more aggressive
        if is_breakout:
            bet_fraction = 0.5 + (0.5 * size_multiplier)
        else:
            bet_fraction = 0.3 * size_multiplier

        # Final safety check: don't bet nothing
        if bet_fraction < 0.05:
            bet_fraction = 0.05

        return side, min(1.0, bet_fraction)

    return None, 0.0