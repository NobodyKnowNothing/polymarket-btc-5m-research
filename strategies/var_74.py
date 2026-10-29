# Var 74 | BT: -1.290 | Live PnL: $-1.29 | Trades: 155
# Desc: Implements "Skew-Microstructure Dislocation Arbitrage (SMDA)," which targets high-alpha opportunities where the Polymarket directional skew (the crowd's bias) is in direct opposition to the microstruc
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, 1.1846420000000002, -1.0, 1.0489065647125244, 0.11476700000000006, 1.6554160000000002, 0.05925399999999992, 1.9791409999999998, -1.0, -1.0, -1.0, 4.245841, -1.0, 1.5191249999999998, -1.0, -0.9485918879508972, -1.0, -1.0, 1.9791409999999998, 2.173394, 0.1769179999999999, 2.073133, -1.0, -1.0, 1.233902, -0.7200189828872681, -1.0, -1.0, 0.03185999393463135, 1.5191249999999998, 2.073133, -1.0, 3.303756, -1.0, 1.5854279999999998, 1.3964720000000002, 1.28551, -1.0, 0.08073200000000003, -1.0, -1.0, 0.08073200000000003, 1.233902, 0.04885600000000001, -1.0, -1.0, -1.0, -0.679844319820404, 1.1375739999999999, 2.39541, -1.0, -0.7866140007972717, -1.0, -1.0, 0.11476700000000006, 3.505625, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -0.7086994647979736, 3.9731249999999996, 2.950496, -0.8999008536338806, -1.0, -1.0, -1.0, -1.0, -0.6428685784339905, -1.0, -0.865031898021698, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, 7.543445, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.06389427185058594, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, 3.9731249999999996, 1.8077519999999998, 2.173394, -1.0, -1.0, 0.05925399999999992, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.09182599999999996, 4.550646, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -0.8879262804985046, -0.904141366481781, -1.0], "live_pnl": -1.2859079837799072, "live_wins": 53, "live_total": 155, "age": 239, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -1.290392962217549, "backtest_acc": 0.5386578208726716, "backtest_n": 3140}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between Polymarket pricing skew and crypto microstructure.
    """
    df = df.copy()

    # 1. Microstructure Signal: Combined direction of OBI and Delta
    # A value of 1 means both OBI and Delta are bullish; -1 means both are bearish.
    df['micro_signal'] = df['mean_delta'] * df['mean_obi']
    df['micro_dir'] = np.sign(df['micro_signal'])

    # 2. Polymarket Skew: The market's implied direction
    # Positive = market favors 'Up'; Negative = market favors 'Down'
    df['pm_skew'] = df['pm_cost_up'] - df['pm_cost_down']
    df['pm_dir'] = np.sign(df['pm_skew'])

    # 3. Flow Intensity: Magnitude of the microstructural pressure
    df['flow_intensity'] = df['mean_delta'].abs() + df['mean_obi'].abs()

    # 4. Volatility-Normalized Micro-Pressure
    df['vol_norm_micro'] = df['micro_signal'] / (df['mean_volatility'] + 1e-6)

    # 5. Price-Strike Proximity (Contextualizing the move)
    df['dist_to_strike_pct'] = df['price_vs_strike_pct']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Skew-Microstructure Dislocation Arbitrage (SMDA)".

    Logic:
    - Dislocation: Microstructure direction != Polymarket skew direction.
      (The crowd is biased one way, but the flow is pushing the other).
      High Alpha: Requires lower EV threshold (0.03) and higher bet size.
    - Consensus: Microstructure direction == Polymarket skew direction.
      (The crowd and flow are aligned).
      Moderate Alpha: Requires higher EV threshold (0.045) and moderate bet size.
    - Neutral: No clear micro-signal or no market skew.
      Low Alpha: Requires very high EV threshold (0.05).
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
    micro_dir = features_row.get('micro_dir', 0.0)
    pm_dir = features_row.get('pm_dir', 0.0)

    # 1. Identify the Regime
    # Dislocation: Micro and PM are moving in opposite directions
    is_dislocation = (micro_dir != 0 and pm_dir != 0 and micro_dir != pm_dir)
    # Consensus: Micro and PM are moving in the same direction
    is_consensus = (micro_dir != 0 and pm_dir != 0 and micro_dir == pm_dir)

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_dislocation:
        # High conviction in the 'correction' of the skew
        req_ev = 0.03
        multiplier = 1.2
    elif is_consensus:
        # Trend-following the consensus
        req_ev = 0.045
        multiplier = 0.7
    else:
        # Noise/Neutral regime
        req_ev = 0.05
        multiplier = 0.4

    # 3. Time-Gating
    # As the window closes, the ability to exploit dislocation diminishes
    if seconds_remaining < 120:
        req_ev *= 1.3
        multiplier *= 0.8

    # 4. Execution
    if best_ev > req_ev:
        # Scale bet size by how much the edge exceeds the required threshold
        # but cap it at the multiplier.
        bet_fraction = multiplier * (best_ev / 0.05)
        return side, min(1.0, bet_fraction)

    return None, 0.0