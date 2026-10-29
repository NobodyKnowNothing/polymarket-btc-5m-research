# Var 81 | BT: -0.949 | Live PnL: $-2.13 | Trades: 132
# Desc: Implements "Microstructure Impact Efficiency (MIE)," which differentiates between "Impulse" regimes (where delta moves price efficiently) and "Absorption" regimes (where aggressive delta is being swal
# Stats: {"live_returns": [1.9791409999999998, 2.173394, 2.073133, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, 1.5191249999999998, 2.073133, 3.303756, -1.0, 1.5854279999999998, 1.3964720000000002, 0.15118699999999996, -1.0, -1.0, 0.08073200000000003, -1.0, -1.0, 0.12663000000000002, 2.39541, -1.0, -1.0, 3.505625, 1.1375739999999999, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, 0.09182599999999996, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, 2.173394, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.9731249999999996, 1.28551, -1.0, 2.39541, -1.0, -1.0, 2.280573, 0.15118699999999996, 0.12663000000000002, 0.13876599999999994, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, 0.16389999999999993, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, 4.550646, 3.1194439999999997, -1.0, 0.09182599999999996, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -2.1282289999999997, "live_wins": 45, "live_total": 132, "age": 264, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": -0.9486173934276847, "backtest_acc": 0.581254724111867, "backtest_n": 2795}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of order flow (how much price moves per unit of delta).
    """
    df = df.copy()

    # 1. Price Momentum (Percentage change in binance price)
    # We group by market_slug to ensure diffs/pct_change don't leak between markets
    df['price_change_pct'] = df.groupby('market_slug')['binance_price'].pct_change()

    # 2. Flow Impact Efficiency: How much does 1 unit of Delta move the price?
    # High = Impulse (Price moves easily, trend is strong)
    # Low = Absorption (Delta is high but price is stuck, potential reversal/exhaustion)
    df['flow_impact'] = df['price_change_pct'].abs() / (df['mean_delta'].abs() + 1e-6)

    # 3. OBI-Delta Synchronization: Do the book and the aggressive flow agree?
    # 1.0 = perfect alignment, -1.0 = perfect opposition
    df['obi_delta_sync'] = df['mean_obi'] * np.sign(df['mean_delta'])

    # 4. Volatility-Adjusted Impact: Is the flow efficiency significant relative to volatility?
    df['vol_adj_impact'] = df['flow_impact'] / (df['mean_volatility'] + 1e-6)

    # 5. Delta Acceleration: Is the aggressive flow increasing or decreasing?
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff()

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
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Impact Efficiency (MIE)".

    Logic:
    - Calculate raw EV for both sides.
    - Identify "Absorption" (Low impact): If delta is high but price isn't moving, 
      the model's directional prediction is likely to face heavy resistance.
    - Identify "Impulse" (High impact): If delta moves price easily, the trend 
      is likely to persist.
    - Apply a 'Sync Penalty': If OBI and Delta are fighting, reduce conviction.
    - Scale bet size based on the 'cleanliness' of the microstructure move.
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
    impact = features_row.get('flow_impact', 0.0)
    sync = features_row.get('obi_delta_sync', 0.0)

    # 1. Multiplier based on Flow Impact (Impulse vs Absorption)
    # Thresholds are calibrated to the scale of pct_change / delta
    multiplier = 1.0
    if impact < 1e-7: # Extreme Absorption: Price is dead despite flow
        multiplier = 0.4
    elif impact > 1e-4: # High Impulse: Price is moving easily with flow
        multiplier = 1.3

    # 2. Sync Penalty: If OBI and Delta are in opposition, reduce confidence
    if sync < 0:
        multiplier *= 0.75

    # 3. Calculate Effective EV
    effective_ev = best_ev * multiplier

    # 4. Dynamic Thresholding
    # Require higher edge as the window closes to avoid volatility spikes
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05
    elif seconds_remaining < 150:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # Scale bet fraction: Aim for 1.0 bet when effective_ev is ~4%
        # We use the multiplier to ensure we bet more on 'clean' moves
        bet_fraction = (best_ev / 0.04) * multiplier
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0