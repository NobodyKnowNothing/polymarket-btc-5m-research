# Var 193 | BT: 0.630 | Live PnL: $-4.75 | Trades: 60
# Desc: Implements "Microstructure Acceleration Synchronization (MAS)," which targets the "Ignition" phase where both order book intent (OBI) and aggressive flow (Delta) are accelerating in unison, while trea
# Stats: {"live_returns": [-0.3, 1.9710447500704464, 1.3946433998837535, -0.9937721968526804, -0.9975515220830573, -0.3, -0.9988732547079531, 0.6841718999999999, -0.3, -0.3, -0.9854541096444386, 0.03811857825609033, -0.3, 0.12466258262514895, 0.5672543999999999, 1.5114856004189585, -0.3, -0.3, 0.037988999999999995, -0.3, 1.868831220012825, -0.99604460662975, -0.9991197806780103, -0.3, -0.3, 0.37017059999999996, 0.04916999999999999, -0.9991829691200288, -0.3, 0.06891903002591881, 0.718623, 0.11353577364841083, -0.9973448770390755, 1.0869053551498518, -0.9990329599154709, 2.163368957963721, -0.9978591347469176, 0.06879018576753393, 0.49662480000000003, 1.3310733005355972, -0.9999093894620762, 0.4368674999999999, -0.3, -0.9965690206530617, 3.505625, 0.17673130075129306, -0.3, -0.3, -0.3, -0.9940151005508171, 1.1183055, -0.9984233546314433, -0.9963181850139091, -0.3, -0.9924152386447853, -0.9675897044823002, -0.9977870524438117, 0.047890779077744194, -0.9990953529776373, -0.996825664409821], "live_pnl": -4.752380960499751, "live_wins": 24, "live_total": 60, "age": 235, "rounds_underwater": 6, "abstain_rounds": 0, "backtest_score": 0.6295023524921077, "backtest_acc": 0.5493761140819964, "backtest_n": 1427}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    second-order derivatives (acceleration) of order book intent and aggressive flow.
    """
    df = df.copy()

    # 1. Calculate First Derivatives (Velocity)
    # We group by market_slug to ensure diffs are calculated within the same 5-min window.
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_vel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 2. Calculate Second Derivatives (Acceleration)
    # Acceleration tells us if the momentum is ramping up or exhausting.
    df['delta_accel'] = df.groupby('market_slug')['delta_vel'].diff().fillna(0)
    df['obi_accel'] = df.groupby('market_slug')['obi_vel'].diff().fillna(0)

    # 3. Acceleration Synchronization Score
    # Positive: Both intent and action are accelerating together (Ignition).
    # Negative: Intent is slowing down while action is speeding up (Friction/Absorption).
    df['accel_sync'] = df['delta_accel'] * df['obi_accel']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noisy acceleration signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.009,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Acceleration Synchronization (MAS)".

    Logic:
    1. Identify Regime:
       - Ignition Regime (accel_sync > 0): Intent and Action are accelerating together.
         This is a high-conviction trend launch. We use a standard hurdle and full bet.
       - Friction Regime (accel_sync < 0): Action is accelerating but Intent is fading.
         This suggests absorption or a liquidity wall. We require a much higher EV 
         and use a reduced bet size to account for the high probability of reversal.
    2. Execution:
       - In Ignition: Scale bet size down if the price is already significantly extended.
       - In Friction: Only trade if the model's predicted edge is massive (compensating for friction).
    """
    # Exit if too little time remains
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    accel_sync = features_row.get('accel_sync', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    price_dist = features_row.get('price_vs_strike_pct', 0.0)

    side = None
    bet_fraction = 0.0

    # --- REGIME 1: IGNITION (Synchronized Acceleration) ---
    if accel_sync > 0:
        # High conviction: follow the model's direction if it aligns with the flow
        if mean_delta > 0 and ev_up > 0.025:
            side = 'Up'
            bet_fraction = 1.0
        elif mean_delta < 0 and ev_down > 0.025:
            side = 'Down'
            bet_fraction = 1.0
        
        # Scale down if we are buying the top/selling the bottom of the move
        if side:
            extension_penalty = 1.0 / (1.0 + abs(price_dist) * 15)
            bet_fraction *= extension_penalty

    # --- REGIME 2: FRICTION (Divergent Acceleration) ---
    elif accel_sync < 0:
        # Low conviction: the flow is hitting a wall. 
        # We only trade if the model's edge is extremely high to offset the friction risk.
        if mean_delta > 0 and ev_up > 0.05:
            side = 'Up'
            bet_fraction = 0.3
        elif mean_delta < 0 and ev_down > 0.05:
            side = 'Down'
            bet_fraction = 0.3

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0