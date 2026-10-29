# Var 85 | BT: 0.152 | Live PnL: $-2.59 | Trades: 94
# Desc: Implements "Lead-Exchange & Absorption Regime Scaling (LEAR)," which dynamically adjusts conviction by detecting if the market is in an "Absorption" state (high flow, no price move) or a "Lead-Exchang
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, 3.505625, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.09182599999999996, -1.0, 1.1846420000000002, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, 2.651597, 7.543445, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.06987699999999997, -1.0, 0.06987699999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, 2.173394, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -2.5876729999999997, "live_wins": 26, "live_total": 94, "age": 176, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": 0.15245038442769265, "backtest_acc": 0.5718905472636816, "backtest_n": 2659}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on
    price-flow efficiency and lead-exchange divergence.
    """
    df = df.copy()

    # 1. Price Movement
    # How much the price has moved from the start of the window
    df['price_move'] = df['binance_price'] - df['strike_price']

    # 2. Price Efficiency (Price change per unit of aggregate flow)
    # Low efficiency = Absorption (high delta, low move); High efficiency = Vacuum/Slippage
    df['price_efficiency'] = df['price_move'] / (df['mean_delta'] + 1e-6)

    # 3. Binance Efficiency (Price change per unit of Binance-specific flow)
    df['binance_efficiency'] = df['price_move'] / (df['binance_delta'] + 1e-6)

    # 4. Efficiency Divergence
    # If Binance efficiency >> aggregate efficiency, Binance is leading the move.
    df['efficiency_diff'] = df['binance_efficiency'] - df['price_efficiency']

    # 5. Delta-OBI Congruence
    df['delta_obi_congruence'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 6. Volatility-Adjusted Flow
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to efficiency ratios."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Exchange & Absorption Regime Scaling (LEAR)".

    Logic:
    - Identify 'Absorption' (Low price efficiency): Scale up if model aligns with flow.
    - Identify 'Lead-Exchange' (High efficiency divergence): Scale up if model aligns with Binance.
    - Identify 'Vacuum' (High price efficiency): Scale down due to slippage/noise risk.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        side_sign = 1
    else:
        best_ev = ev_down
        side = 'Down'
        side_sign = -1

    # Extract engineered features
    eff = features_row.get('price_efficiency', 0.0)
    eff_diff = features_row.get('efficiency_diff', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    binance_delta = features_row.get('binance_delta', 0.0)
    
    regime_mult = 1.0

    # 1. Absorption Regime Detection
    # Price is not moving despite significant aggressive flow
    if abs(eff) < 0.01 and abs(mean_delta) > 0.1:
        if side_sign == np.sign(mean_delta):
            regime_mult = 1.4  # Breakout potential: High conviction
        else:
            regime_mult = 0.4  # Fighting the wall: Low conviction

    # 2. Lead-Exchange Detection
    # Binance is moving price significantly more efficiently than the market average
    elif abs(eff_diff) > 0.05:
        if side_sign == np.sign(binance_delta):
            regime_mult = 1.3  # Binance is leading: High conviction
        else:
            regime_mult = 0.6  # Binance is opposing: Low conviction

    # 3. Vacuum/Slippage Detection
    # Price is moving too easily (low liquidity/high slippage)
    if abs(eff) > 0.5:
        regime_mult *= 0.5

    # Calculate Effective EV
    effective_ev = best_ev * regime_mult

    # 4. Dynamic Thresholding
    # Tighten requirements as the window closes to avoid noise
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 150:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # Scale bet fraction: target 4% edge for 1.0 bet
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0