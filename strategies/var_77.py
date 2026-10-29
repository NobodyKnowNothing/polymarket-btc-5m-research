# Var 77 | BT: -0.367 | Live PnL: $-1.81 | Trades: 127
# Desc: Targets high-conviction breakouts by identifying "coiled" market states (low volatility and low exchange dispersion) followed by a significant, consensus-driven surge in order flow (delta z-score).
# Stats: {"live_returns": [1.6554160000000002, 0.04971921443939209, 1.9791409999999998, -1.0, -1.0, 0.03413569927215576, 4.245841, -1.0, 1.5053112506866455, -1.0, 0.03413569927215576, -1.0, -0.888512372970581, 1.9791409999999998, 2.173394, 2.073133, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, 1.5191249999999998, 2.073133, -0.750925600528717, -0.750925600528717, -1.0, 3.303756, -1.0, 1.5854279999999998, 1.1473407745361328, -1.0, -1.0, 1.28551, -1.0, -1.0, -0.9932419061660767, -1.0, -1.0, -1.0, 1.233902, 0.03413569927215576, -1.0, -1.0, 1.1375739999999999, -1.0, 2.39541, -1.0, -1.0, -1.0, 3.505625, 2.950496, 2.173394, 0.1062052845954895, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.046170711517334, -0.7569824457168579, -1.0, -0.9656476378440857, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -0.7090590000152588, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0190811157226562, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7587571740150452, 4.550646, 3.1194439999999997, -1.0, -1.0, 0.05682307481765747, -1.0, -0.7587571740150452, -0.851053774356842, 2.39541, 0.8988591432571411, -0.851053774356842, -1.0, 1.4562249999999999, -1.0, -0.851053774356842, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -0.7015623450279236, -1.0, -1.0], "live_pnl": -1.8067366480827332, "live_wins": 43, "live_total": 127, "age": 198, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.36748299935225054, "backtest_acc": 0.5637755102040817, "backtest_n": 2913}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    detecting "coiled" volatility regimes and sudden order-flow breakouts.
    """
    df = df.copy()

    # 1. Price Momentum (per market)
    df['price_momentum'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Volatility Z-Score: Detects "coiled" states (low vol relative to recent window)
    # We use a rolling window of 15 snapshots (~2.5 mins)
    vol_roll = df.groupby('market_slug')['mean_volatility']
    df['vol_rolling_mean'] = vol_roll.transform(lambda x: x.rolling(window=15, min_periods=1).mean())
    df['vol_rolling_std'] = vol_roll.transform(lambda x: x.rolling(window=15, min_periods=1).std())
    df['vol_zscore'] = (df['mean_volatility'] - df['vol_rolling_mean']) / (df['vol_rolling_std'] + 1e-6)

    # 3. Delta Z-Score: Detects sudden surges in aggressive flow
    delta_roll = df.groupby('market_slug')['sum_delta']
    df['delta_rolling_mean'] = delta_roll.transform(lambda x: x.rolling(window=15, min_periods=1).mean())
    df['delta_rolling_std'] = delta_roll.transform(lambda x: x.rolling(window=15, min_periods=1).std())
    df['delta_zscore'] = (df['sum_delta'] - df['delta_rolling_mean']) / (df['delta_rolling_std'] + 1e-6)

    # 4. OBI Agreement: How unified is the order book across exchanges?
    # High agreement = low spread.
    df['obi_agreement'] = 1.0 - df['obi_spread'].clip(0, 1)

    # 5. Price Velocity: Momentum normalized by volatility
    df['price_velocity'] = df['price_momentum'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 
                    'vol_rolling_mean', 'vol_rolling_std', 
                    'delta_rolling_mean', 'delta_rolling_std']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to z-score noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Coiled-Spring Momentum Capture (CSMC)".

    Logic:
    - Coiled State: Low volatility (vol_zscore < -0.5) and high OBI agreement.
    - Trigger: A significant spike in aggressive flow (abs(delta_zscore) > 1.5).
    - Goldilocks Zone: Coiled + Trigger + Agreement = Very low EV threshold (0.02).
    - Standard Breakout: Trigger + Agreement = Moderate EV threshold (0.035).
    - Noise: Everything else = High EV threshold (0.045) and reduced sizing.
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
    v_z = features_row.get('vol_zscore', 0.0)
    d_z = features_row.get('delta_zscore', 0.0)
    o_a = features_row.get('obi_agreement', 0.0)

    # 1. Identify Market Regimes
    is_coiled = v_z < -0.5
    is_trigger = abs(d_z) > 1.5
    is_consensus = o_a > 0.7

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_coiled and is_trigger and is_consensus:
        # The "Spring" is releasing: high conviction, low entry barrier
        req_ev = 0.02
        multiplier = 1.2
    elif is_trigger and is_consensus:
        # Standard momentum breakout
        req_ev = 0.035
        multiplier = 1.0
    else:
        # Noise or non-consensus: require high edge to justify risk
        req_ev = 0.045
        multiplier = 0.6

    # 3. Time-Gating
    # As the window closes, the "spring" effect diminishes and noise increases
    if seconds_remaining < 60:
        req_ev *= 1.5
        multiplier *= 0.7

    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% baseline
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0