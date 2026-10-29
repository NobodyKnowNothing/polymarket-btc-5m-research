# Var 209 | BT: 0.304 | Live PnL: $-6.73 | Trades: 60
# Desc: Implements "Microstructure Efficiency Divergence (MED)," which targets the relationship between flow intensity (Delta/OBI) and price impact to identify "Coiled Springs" (absorption), "Breakouts" (high
# Stats: {"live_returns": [0.10130400000000006, 1.5126784000000002, -0.8, 0.06458560000000002, -0.8, -0.8, -0.8, 0.09181360000000005, 0.07346079999999999, 0.030939999999999968, 0.055901599999999996, 0.0825343999999999, 0.030939999999999968, 0.09181360000000005, -0.8, 0.874044, -0.8, 1.7387152000000003, -0.8, 0.04740319999999998, 0.055901599999999996, -0.8, 1.3243328000000003, -0.8, -0.8, 0.07346079999999999, -0.8, 0.022964799999999896, 2.8045, -0.8, -0.8, -0.8, 0.128552, 2.9821480000000005, -0.8, -0.8, -0.8, -0.8, 0.03908480000000003, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, 0.09181360000000005, 1.2152999999999998, 0.1311199999999999, 0.10130400000000006, -0.8, -0.8, -0.8, 1.5126784000000002, 0.055901599999999996, -0.8, 0.09181360000000005, 1.8244584, -0.8, -0.8, 0.022964799999999896], "live_pnl": -6.725566399999997, "live_wins": 30, "live_total": 60, "age": 213, "rounds_underwater": 25, "abstain_rounds": 0, "backtest_score": 0.3044308478614784, "backtest_acc": 0.5718020216103172, "backtest_n": 1395}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement relative to order flow (Delta and OBI).
    """
    df = df.copy()

    # 1. Calculate Price Change (within each market window)
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Efficiency Metrics: How much price moves per unit of flow
    # High efficiency = small flow moves price a lot (Liquidity Vacuum)
    # Low efficiency = huge flow moves price a little (Absorption/Wall)
    df['delta_eff'] = df['price_change'] / (df['mean_delta'].abs() + 1e-6)
    df['obi_eff'] = df['price_change'] / (df['mean_obi'].abs() + 1e-6)

    # 3. Directional Signs
    df['delta_sign'] = np.sign(df['mean_delta'])
    df['obi_sign'] = np.sign(df['mean_obi'])

    # 4. Interaction: Efficiency Divergence
    # Positive if Delta is more efficient than OBI, negative if OBI is more efficient
    df['eff_divergence'] = df['delta_eff'] - df['obi_eff']

    # 5. Volatility-Adjusted Flow
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on structural signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Efficiency Divergence (MED)".

    Logic:
    1. Absorption (The Wall): Low Delta Efficiency + High OBI Efficiency + Sign Mismatch.
       Aggressive sellers/buyers are hitting a passive wall. Predict the wall's direction.
    2. Momentum (The Breakout): High Delta Efficiency + High OBI Efficiency.
       Flow and Intent are both successfully moving the price.
    3. Exhaustion (The Fade): Extreme Delta Efficiency + Low OBI Efficiency.
       Price has moved significantly on tiny flow, suggesting a liquidity vacuum/blow-off.
    """
    # Time safety
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    d_eff = features_row.get('delta_eff', 0.0)
    o_eff = features_row.get('obi_eff', 0.0)
    d_sign = features_row.get('delta_sign', 0.0)
    o_sign = features_row.get('obi_sign', 0.0)
    
    side = None
    bet_fraction = 0.0
    hurdle = 0.03  # Default hurdle

    # --- REGIME 1: ABSORPTION (The Coiled Spring) ---
    # Price isn't moving despite aggressive delta, but OBI is driving price.
    # This suggests a massive passive wall is absorbing the aggression.
    if abs(d_eff) < 0.002 and abs(o_eff) > 0.002 and d_sign != 0 and o_sign != 0:
        if d_sign != o_sign:
            # If delta is negative (selling) but OBI is positive (bids), price will likely bounce UP
            side = 'Up' if o_sign > 0 else 'Down'
            hurdle = 0.045 # Higher hurdle for absorption reversals
            bet_fraction = 0.7

    # --- REGIME 2: MOMENTUM (The Breakout) ---
    # Both aggressive flow and passive intent are aligned and moving the price efficiently.
    elif abs(d_eff) > 0.01 and abs(o_eff) > 0.01:
        if d_sign > 0:
            side = 'Up'
        elif d_sign < 0:
            side = 'Down'
        hurdle = 0.025
        bet_fraction = 0.8

    # --- REGIME 3: EXHAUSTION (The Fade) ---
    # Price moved a lot on tiny delta, but there's no OBI support.
    elif abs(d_eff) > 0.05 and abs(o_eff) < 0.002:
        side = 'Down' if d_sign > 0 else 'Up'
        hurdle = 0.035
        bet_fraction = 0.5

    # --- EXECUTION ---
    if side == 'Up':
        if ev_up > hurdle and ev_up > ev_down:
            return 'Up', bet_fraction
    elif side == 'Down':
        if ev_down > hurdle and ev_down > ev_up:
            return 'Down', bet_fraction

    return None, 0.0