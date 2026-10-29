# Var 191 | BT: -0.357 | Live PnL: $-5.68 | Trades: 60
# Desc: Implements "Microstructure Impact Efficiency (MIE) Regime Switching," which identifies "Absorption" (high delta with low price impact and opposing OBI) for reversals and "Impulse" (high delta with hig
# Stats: {"live_returns": [-0.5, -0.5, -0.5, 1.197705, 1.1402865, -0.5, 0.8647035000000001, -0.5, -0.5, -0.5, 0.014352999999999949, 0.1234845, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.7281124999999999, -0.5, 0.02962699999999996, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 1.1402865, -0.5, 0.05738350000000003, 0.05738350000000003, -0.5, -0.5, -0.5, 1.1402865, 0.5923210000000001, 0.945424, -0.5, 0.616951, 0.6698185000000001, -0.5, -0.5, -0.5, 0.5923210000000001, -0.5, 0.6982360000000001, -0.5, -0.5, -0.5, 0.8647035000000001, 2.275323, -0.5, -0.5, 0.5687869999999999], "live_pnl": -5.682502999999999, "live_wins": 20, "live_total": 60, "age": 380, "rounds_underwater": 51, "abstain_rounds": 0, "backtest_score": -0.3565273453274283, "backtest_acc": 0.5665289998204346, "backtest_n": 889}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    efficiency of price movement relative to aggressive order flow (Delta).
    """
    df = df.copy()

    # 1. Calculate price movement magnitude within the market window
    # We use the change in the percentage distance from the strike price.
    df['price_move'] = df.groupby('market_slug')['price_vs_strike_pct'].diff().fillna(0)

    # 2. Impact Efficiency: How much Delta is required to move the price?
    # High efficiency = high delta, low price move (Absorption)
    # Low efficiency = high delta, high price move (Impulse/Momentum)
    # We clip to 100 to prevent extreme outliers from destabilizing XGBoost.
    df['impact_efficiency'] = (df['mean_delta'].abs() / (df['price_move'].abs() + 1e-7)).clip(0, 100)

    # 3. Order-Flow/Intent Alignment
    # Positive: Delta and OBI are in the same direction (supporting a move)
    # Negative: Delta and OBI are opposing (fighting a move/absorption)
    df['obi_delta_alignment'] = df['mean_obi'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to handle the noisy efficiency ratio."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Impact Efficiency (MIE) Regime Switching".

    Logic:
    1. Identify Regime:
       - Absorption Regime: High impact_efficiency + negative obi_delta_alignment.
         Aggressive flow is being met by a heavy order book wall. Expect reversal.
       - Impulse Regime: Low impact_efficiency + positive obi_delta_alignment.
         Aggressive flow is moving price easily and is supported by the book. Expect continuation.
    2. Execution:
       - In Absorption: Bet AGAINST the direction of the aggressive Delta.
       - In Impulse: Bet WITH the direction of the aggressive Delta.
    """
    # Exit if too little time remains to capture alpha or if edge is too thin
    if seconds_remaining < 60:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Minimum hurdle for any trade
    if max(ev_up, ev_down) < 0.02:
        return None, 0.0

    # Extract engineered features
    eff = features_row.get('impact_efficiency', 0.0)
    alignment = features_row.get('obi_delta_alignment', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)

    side = None
    bet_fraction = 0.0

    # --- REGIME 1: ABSORPTION (High Delta, Low Price Move, Opposing OBI) ---
    # Threshold for efficiency: high value means delta is "wasted"
    if eff > 15.0 and alignment < -0.01:
        if mean_delta > 0 and ev_down > 0.02:
            # Aggressive buying is being absorbed -> Bet Down
            side = 'Down'
            bet_fraction = 0.5 
        elif mean_delta < 0 and ev_up > 0.02:
            # Aggressive selling is being absorbed -> Bet Up
            side = 'Up'
            bet_fraction = 0.5

    # --- REGIME 2: IMPULSE (High Delta, High Price Move, Aligned OBI) ---
    # Threshold for efficiency: low value means delta is "efficient"
    elif eff < 0.8 and alignment > 0.01:
        if mean_delta > 0 and ev_up > 0.02:
            # Aggressive buying is driving price -> Bet Up
            side = 'Up'
            bet_fraction = 0.8
        elif mean_delta < 0 and ev_down > 0.02:
            # Aggressive selling is driving price -> Bet Down
            bet_fraction = 0.8
            side = 'Down'

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0