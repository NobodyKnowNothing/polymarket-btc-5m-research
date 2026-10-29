# Var 72 | BT: 0.304 | Live PnL: $-1.03 | Trades: 408
# Desc: Implements "Microstructure-Validated Expected Value (MVEV)," which scales the model's expected value by a congruence factor (alignment between predicted direction and actual order flow/book) and a dis
# Stats: {"live_returns": [-1.0, 0.08402681350708008, 2.173394, 3.1194439999999997, -1.0, -1.0, 2.950496, -1.0, -0.939762532711029, -1.0, -1.0, 0.03492683172225952, 1.6554160000000002, 0.08073200000000003, 2.39541, -1.0, -1.0, -0.8691785931587219, -1.0, 1.8077519999999998, -1.0, 1.2911014556884766, 0.053164005279541016, 0.11476700000000006, 1.6554160000000002, 1.9791409999999998, -1.0, 0.15118699999999996, 4.245841, -1.0, 1.5191249999999998, 1.9791409999999998, 2.173394, 2.073133, -1.0, -1.0, 1.233902, -1.0, -1.0, 1.5191249999999998, 2.073133, -1.0, -1.0, 3.303756, -1.0, 0.10316799999999993, 0.10316799999999993, 1.5854279999999998, 1.3964720000000002, 0.15118699999999996, -0.8640245795249939, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, 3.505625, 2.950496, 3.505625, 2.173394, 1.890848, -1.0, -1.0, -1.0, 2.280573, -0.8237040042877197, 1.28551, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.9731249999999996, 2.950496, -0.797151505947113, -1.0, -1.0, -0.8949015736579895, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -0.8479637503623962, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.06987699999999997, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, 1.8077519999999998, 2.173394, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8831387758255005, -1.0, -1.0, 0.12663000000000002, -1.0, -1.0, -1.0, -1.0, 3.9731249999999996, 1.0127955675125122, -1.0, 2.39541, 1.0864334106445312, -1.0, -1.0, -1.0, -0.9610146284103394, 2.280573, 0.042167484760284424, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9604415893554688, -1.0, -1.0, 1.4011807441711426, 0.9049400091171265, -1.0, -1.0, 0.10477322340011597, -1.0, -1.0, -1.0, 1.9791409999999998, 0.15118699999999996, -1.0, 3.303756, 0.9673321843147278, 1.890848, -1.0, 0.12663000000000002, -1.0, 1.5191249999999998, 1.1077659130096436, 1.1846420000000002, 1.3396370000000002, 0.12364464998245239, -1.0, -0.8228538036346436, 2.280573, -0.9557915329933167, -1.0, 1.4562249999999999, 0.9434523582458496, -0.7639146447181702, 1.6554160000000002, -1.0, -1.0, -0.9903765320777893, 4.245841, 0.13876599999999994, 2.795066, 0.06987699999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8022769689559937, 6.238865, 3.727685, 0.08073200000000003, -0.9663105010986328, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 0.09911000728607178, -1.0, -1.0, 2.950496, 1.2165977954864502, 3.9731249999999996, 0.08874201774597168, -1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, 4.893558, 2.39541, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, 0.07364696264266968, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.3191636800765991, -1.0, -1.0, -1.0, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -0.9116024971008301, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -0.9158337712287903, 3.727685, 2.795066, 1.9791409999999998, -1.0, 1.28551, 2.795066, 3.727685, 3.1194439999999997, 1.3396370000000002, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8105865120887756, -1.0, 2.795066, -1.0, 2.280573, -1.0, -1.0, -0.9958915710449219, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, 0.9779424667358398, -0.925898015499115, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7964430451393127, 1.0141526460647583, 2.39541, 1.049452, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, -1.0, 3.727685, -1.0, 1.2228904962539673, -0.7984435558319092, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9004867672920227, -1.0, -1.0, -1.0, -1.0, -0.8607397079467773, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.0802431106567383, -1.0, -1.0, -0.9087669849395752, -1.0, 0.08002614974975586, 1.8077519999999998, -1.0, -0.9765871167182922, 1.7294070000000001, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, -0.8444673418998718, -1.0, -1.0, -1.0, 1.2116113901138306, -1.0, -1.0, -0.9956346154212952, -1.0, 2.950496, 0.8359411954879761, -1.0, -1.0, -1.0, 1.0406465530395508, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.0317573547363281, "live_wins": 143, "live_total": 408, "age": 811, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.3043508750146173, "backtest_acc": 0.544355875546413, "backtest_n": 2596}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure congruence and cross-exchange price stability.
    """
    df = df.copy()

    # 1. Delta-OBI Congruence: Do the aggressive flow and the order book agree?
    # 1.0 = perfect alignment, -1.0 = perfect opposition, 0 = neutral.
    df['delta_obi_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 2. Price Dispersion: How much do exchanges disagree on the price?
    # High dispersion often precedes high volatility or indicates low-liquidity noise.
    df['price_dispersion_pct'] = df['price_std'] / (df['binance_price'] + 1e-6)

    # 3. Funding-Flow Interaction: Is the aggressive flow aligned with perp funding?
    # High positive = aggressive buying in a long-heavy market (potential squeeze).
    df['funding_flow_interaction'] = df['mean_funding'] * df['mean_delta']

    # 4. Volatility-Normalized Delta: Is the flow significant relative to current volatility?
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust microstructure signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Validated Expected Value (MVEV)".

    Logic:
    - Calculate raw EV for both sides.
    - Apply a 'Congruence Factor': If the model's best side contradicts the 
      sign of 'mean_delta' or 'mean_obi', reduce the effective EV (potential trap).
    - Apply a 'Dispersion Penalty': Scale down the bet if cross-exchange 
      price dispersion is high (signal noise).
    - Time-gate: Increase the required EV threshold as the market window closes.
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
    delta_sign = np.sign(features_row.get('mean_delta', 0.0))
    obi_sign = np.sign(features_row.get('mean_obi', 0.0))
    dispersion_pct = features_row.get('price_dispersion_pct', 0.0)

    # 1. Congruence Factor (0.5 to 1.0)
    # If we want to go 'Up', we want delta and OBI to be positive.
    # If they are negative, the model is betting against the immediate flow.
    congruence = 1.0
    if side == 'Up':
        if delta_sign < 0 or obi_sign < 0:
            congruence = 0.5
    else: # side == 'Down'
        if delta_sign > 0 or obi_sign > 0:
            congruence = 0.5

    # 2. Dispersion Penalty (scales down as dispersion increases)
    # We use a multiplier that decays as price_dispersion_pct grows.
    dispersion_penalty = 1.0 / (1.0 + (dispersion_pct * 1000.0))

    # 3. Calculate Effective EV
    effective_ev = best_ev * congruence * dispersion_penalty

    # 4. Dynamic Thresholding
    # Base threshold is 0.03. Tighten as time runs out to avoid last-second noise.
    base_threshold = 0.03
    if seconds_remaining < 45:
        base_threshold = 0.05
    elif seconds_remaining < 120:
        base_threshold = 0.04

    if effective_ev > base_threshold:
        # Scale bet fraction based on how much the effective EV exceeds the threshold
        # Target a 4% edge for a full 1.0 bet.
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0