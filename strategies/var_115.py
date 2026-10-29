# Var 115 | BT: -0.532 | Live PnL: $-1.27 | Trades: 109
# Desc: Implements "Lead-Lag & Absorption Gating (LLAG)," which targets dislocations where Binance leads the market (high lead-lag) while filtering out high-delta/low-OBI "absorption" regimes that signal exha
# Stats: {"live_returns": [-1.0, 2.795066, 1.890848, 1.890848, -1.0, -1.0, -1.0, -1.0, 0.1282358169555664, -1.0, 1.0842558145523071, 2.950496, 1.1846420000000002, -1.0, -0.8540884256362915, 0.5254305005073547, -0.7521483302116394, -1.0, 0.11476700000000006, 0.032076120376586914, 1.1846420000000002, -1.0, -1.0, 0.06111079454421997, -1.0, -0.9035512804985046, 1.5191249999999998, 2.073133, -1.0, -1.0, -0.5793243646621704, -1.0, -0.7064461708068848, -0.750228762626648, -1.0, -1.0, -0.750228762626648, 6.238865, 3.727685, -1.0, -0.6716588735580444, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -0.8932286500930786, 0.11476700000000006, -1.0, -1.0, 0.0456387996673584, -1.0, 2.950496, 1.4312959909439087, 3.9731249999999996, -1.0, -1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -0.5777010321617126, 4.893558, -1.0, 2.39541, -0.8082109689712524, -1.0, -1.0, -1.0, 2.651597, -1.0, 0.025201082229614258, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.6049409508705139, -1.0, -1.0, 0.046619713306427, -1.0, -0.8299890756607056, -1.0, -0.7860374450683594, 1.4562249999999999, -1.0, -1.0, -1.0, 0.03314614295959473, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.2693681716918945, "live_wins": 37, "live_total": 109, "age": 182, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.5324747305018751, "backtest_acc": 0.5101329265635215, "backtest_n": 3654}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Binance lead-lag dislocations and microstructure absorption.
    """
    df = df.copy()

    # 1. Lead-Lag Features: How much is Binance deviating from the global mean?
    # High positive = Binance is much more bullish than the other exchanges.
    df['binance_lead_obi'] = df['binance_obi'] - df['mean_obi']
    df['binance_lead_delta'] = df['binance_delta'] - df['mean_delta']

    # 2. Absorption Index: High aggressive delta with little OBI movement 
    # suggests limit orders are absorbing the flow (potential exhaustion).
    # We add a small constant to prevent division by zero.
    df['absorption_index'] = df['mean_delta'].abs() / (df['mean_obi'].abs() + 0.1)

    # 3. Price Spread Velocity: Is the arbitrage/dispersion tightening or widening?
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 4. Momentum of the Lead Venue
    df['binance_obi_mom'] = df.groupby('market_slug')['binance_obi'].diff().fillna(0)
    df['binance_delta_mom'] = df.groupby('market_slug')['binance_delta'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to capture structural dislocations."""
    return {
        'n_estimators': 800,
        'max_depth': 3,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 5.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Lag & Absorption Gating (LLAG)".

    Logic:
    - Lead-Lag Breakout: Binance OBI and Delta are strongly aligned and 
      diverging from the global mean. (Low threshold, high bet).
    - Absorption/Exhaustion: High delta but low OBI movement. (High threshold, low bet).
    - Conflict: Binance OBI and Delta are moving in opposite directions. (High threshold).
    - Neutral: Standard EV threshold.
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
    lead_obi = features_row.get('binance_lead_obi', 0.0)
    lead_delta = features_row.get('binance_lead_delta', 0.0)
    abs_idx = features_row.get('absorption_index', 0.0)
    
    # 1. Identify Market Regimes
    # Lead-Lag: Binance is significantly leading the direction of the market
    is_lead_lag_up = lead_obi > 0.25 and lead_delta > 1.0
    is_lead_lag_down = lead_obi < -0.25 and lead_delta < -1.0
    is_lead_lag = is_lead_lag_up or is_lead_lag_down

    # Absorption: Aggressive flow is hitting a wall (high delta, low OBI)
    is_absorption = abs_idx > 4.0

    # Conflict: Binance OBI and Delta are fighting each other
    is_conflict = (lead_obi * lead_delta) < 0

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_lead_lag and not is_absorption:
        # High conviction: Binance is leading a clear, non-absorbed move
        req_ev = 0.020
        multiplier = 1.0
    elif is_absorption or is_conflict:
        # High risk: Market is either absorbing flow or Binance is conflicted
        req_ev = 0.050
        multiplier = 0.4
    else:
        # Neutral: Standard regime
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating: Reduce confidence as the window closes
    if seconds_remaining < 45:
        req_ev *= 1.3
        multiplier *= 0.7

    # 4. Execution
    if best_ev > req_ev:
        # Scale bet size based on how much the edge exceeds the threshold
        # and the magnitude of the multiplier
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0