# Var 75 | BT: -0.136 | Live PnL: $-1.88 | Trades: 144
# Desc: Implements "Spot-Polymarket Latency Capture (SPLC)," which targets high-conviction trades where spot microstructure momentum (price and delta velocity) is accelerating but Polymarket's directional ske
# Stats: {"live_returns": [-1.0, -1.0, 1.8077519999999998, -1.0, 0.9651845693588257, 1.3396370000000002, 0.11476700000000006, -0.7090973854064941, 1.6554160000000002, 1.9791409999999998, -1.0, -1.0, 0.05925399999999992, 4.245841, -1.0, 1.5191249999999998, -1.0, -1.0, -0.9227240681648254, 0.06987699999999997, -1.0, 1.9791409999999998, 2.173394, 2.073133, -1.0, -1.0, -0.9485793709754944, 1.233902, -1.0, -1.0, -1.0, 1.5191249999999998, 2.073133, -1.0, 0.16389999999999993, 3.303756, -1.0, 0.10316799999999993, 1.5854279999999998, 1.3964720000000002, 0.15118699999999996, 1.28551, -1.0, -1.0, 1.233902, 0.04885600000000001, -1.0, -1.0, -0.963901937007904, 2.39541, -1.0, -1.0, -1.0, 3.505625, 0.8956322073936462, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0872039794921875, -0.9177488684654236, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, 1.28551, 0.04885600000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.038675000000000015, -0.9061575531959534, 2.651597, -0.8240434527397156, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.049452, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.05925399999999992, 1.28551, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -0.7038239240646362, -1.0, -1.0, -1.0], "live_pnl": -1.8835041522979736, "live_wins": 52, "live_total": 144, "age": 215, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.13595192862025565, "backtest_acc": 0.5723029839326702, "backtest_n": 3093}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between spot momentum and Polymarket pricing skew.
    """
    df = df.copy()

    # 1. Velocity Features: Rate of change in price and aggressive flow
    # We group by market_slug to ensure diffs are calculated within the same 5-min window
    df['price_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_vel'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)

    # 2. Impulse Product: Captures the 'energy' of a move. 
    # Positive = Price and Delta moving together (Trend); Negative = Price and Delta diverging (Absorption)
    df['impulse_prod'] = df['price_vel'] * df['delta_vel']

    # 3. Polymarket Skew: The current market sentiment expressed in costs
    df['pm_skew'] = df['pm_cost_up'] - df['pm_cost_down']

    # 4. Volatility-Adjusted Impulse: Normalizes the impulse by the current regime
    df['vol_adj_impulse'] = df['impulse_prod'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Spot-Polymarket Latency Capture (SPLC)".

    Logic:
    - Identifies "Latency Opportunities": When spot momentum (impulse_prod) is strong 
      but moving in the opposite direction of the Polymarket skew.
    - Trend Trades: When momentum and skew are aligned, we require a higher EV 
      to avoid "crowded" trades that are already priced in.
    - Latency Trades: When momentum and skew diverge, we lower the EV threshold 
      to capture the move before the market corrects.
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
    impulse = features_row.get('impulse_prod', 0.0)
    skew = features_row.get('pm_skew', 0.0)
    
    # 1. Detect Divergence (Latency Opportunity)
    # A divergence occurs if the sign of the price/delta impulse is opposite to the sign of the skew
    # and the impulse is significant enough to not be noise.
    is_divergent = False
    if abs(impulse) > 0.0001: # Minimum impulse threshold
        if (np.sign(impulse) * np.sign(skew)) < 0:
            is_divergent = True

    # 2. Regime-Based Thresholding
    if is_divergent:
        # Latency Opportunity: The move is happening in spot but hasn't hit Polymarket yet.
        # We are aggressive here.
        req_ev = 0.020
        multiplier = 1.2
    else:
        # Trend/Crowded Trade: The move is already reflected in the skew.
        # We require more edge to ensure we aren't buying the top.
        req_ev = 0.040
        multiplier = 0.8

    # 3. Time-Gating: As the window closes, uncertainty increases.
    if seconds_remaining < 45:
        req_ev *= 1.3
        multiplier *= 0.7

    # 4. Execution
    if best_ev > req_ev:
        # Scale bet size by edge magnitude, capped at 1.0
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, bet_fraction)

    return None, 0.0