# Var 194 | BT: 0.666 | Live PnL: $11.59 | Trades: 139
# Desc: Implements "Liquidation-Cascade Squeeze (LCS)," which targets explosive price movements triggered by forced liquidations by identifying the confluence of extreme perp funding and aggressive delta in t
# Stats: {"live_returns": [0.2283624230769231, 0.8862021361525375, 0.6982360000000001, -0.797593969, 0.011476700000000006, -0.76890330425, -0.1, 0.014653594865596475, -0.7893775245, 0.04591299999999998, 1.458550340845544, -0.8336543867499999, 1.1717757774937496, -0.1, -0.785422906, -0.995734057, -0.11538461538461539, -0.8517667225, 0.019337500000000007, 0.0863923476345313, -0.5, 0.8239672250305874, -0.5, 2.173394, -0.11538461538461539, 0.09511443310887002, 0.14592649731420948, 1.221404362317958, -0.917751112, -0.5, -0.7880533, 3.3435714827450003, 0.08845899999999995, -0.11538461538461539, -0.5, -0.74491876, -0.733962832, 0.003867499999999996, -0.1, -0.777767476, -0.7504048959999999, 1.2150570129179998, 0.012663000000000008, 0.11001687974838392, -0.73330794275, 0.08654505585874583, 0.21817476923076926, -0.8912512155, -0.75357098225, 0.05738350000000003, -0.1, -1.0, -0.1, -0.11538461538461539, 0.07559349999999998, 9.2346277, -0.773582530125, 0.05230603814809731, 0.08845899999999995, -0.7954361725, -0.796367018125, -0.5, 0.616951, 0.12309920159000631, -0.1, -0.755527391125, 0.8718646460186817, 0.8647035000000001, -0.5, -0.1, -0.8046074514999999, 0.1266834344443878, 0.616951, -0.5, 0.09182599999999996, 0.10104974704668379, -0.7687365504999999, -0.1, 3.155014457336015, 1.118679712698977, -0.1, 2.0696502145221065, 0.0982533533263682, -0.756725362, -0.1, 0.01603634590082048, -1.0, 2.114644507960747, -0.5, 0.07943339590950393, -0.787277033125, -0.8109640914999999, -0.7908110653749999, -0.1, -0.1, -0.1, 0.003867499999999996, 1.407948957366836, 0.1377664166038569, 0.09182599999999996, -0.1, -0.1, -0.1, 2.2044725329764505, -1.0, 0.04885600000000001, -0.1, 3.0483637869749995, -1.0, -0.823317077, -0.1, -0.1, -0.7566112939999999, -0.7734950304999999, -0.7389364294999999, 0.21733940000000004, -0.8043109875, 1.06966665775716, -0.11538461538461539, -0.7882982525, 2.157642830272396, -0.5, 0.10000763803800006, 0.011903999999999998, -0.11538461538461539, -0.82658048, -0.1, -0.7396590944999999, -0.1, -0.8200764949999999, 0.013876599999999989, 0.08254516212467189, 3.429656162498649, 1.83139968463695, -0.1, 0.1456225, 0.028705999999999898, -0.8601844025, -0.1], "live_pnl": 11.586485834811084, "live_wins": 61, "live_total": 139, "age": 842, "rounds_underwater": 0, "abstain_rounds": 11, "backtest_score": 0.6663936275619058, "backtest_acc": 0.5673352435530086, "backtest_n": 1232}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    liquidation-triggering dynamics (Funding-Delta confluence).
    """
    df = df.copy()

    # 1. Delta Velocity (Acceleration of aggressive flow)
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Funding-Delta Confluence (Detecting if flow is fighting the leverage)
    # Negative value indicates potential squeeze (e.g., positive funding + negative delta)
    df['funding_delta_confluence'] = df['mean_funding'] * df['mean_delta']

    # 3. OBI-Delta Alignment (Are intent and action in sync?)
    df['obi_delta_alignment'] = df['mean_obi'] * df['mean_delta']

    # 4. Volatility-Normalized Delta (Signal strength relative to noise)
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

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
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidation-Cascade Squeeze (LCS)" logic.

    Logic:
    1. Identify Squeeze Regime:
       - A squeeze occurs when 'funding_delta_confluence' is significantly negative.
       - This means aggressive delta is moving AGAINST the dominant perp position.
       - If mean_delta < 0 and funding > 0 -> Long Squeeze (Bet Down).
       - If mean_delta > 0 and funding < 0 -> Short Squeeze (Bet Up).
    2. Execution:
       - In Squeeze: Higher conviction, larger bet size, as liquidations create cascades.
       - In Trend: Standard EV-based entry, but scaled down by volatility to avoid 'buying the top'.
    """
    # Exit if too little time remains
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    funding_delta_conf = features_row.get('funding_delta_confluence', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_funding = features_row.get('mean_funding', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)
    obi_delta_align = features_row.get('obi_delta_alignment', 0.0)

    side = None
    bet_fraction = 0.0

    # --- REGIME 1: LIQUIDATION SQUEEZE (High Intensity) ---
    # A squeeze is signaled by opposing funding and delta.
    # We use a threshold to ensure the signal is strong enough to overcome costs.
    squeeze_threshold = -0.00002 
    
    if funding_delta_conf < squeeze_threshold:
        # Potential Long Squeeze (Selling into long positions)
        if mean_delta < 0 and ev_down > 0.015:
            side = 'Down'
            # Scale bet by intensity of the squeeze
            intensity = min(1.0, abs(funding_delta_conf) * 5000)
            bet_fraction = 0.7 + (0.3 * intensity)
            
        # Potential Short Squeeze (Buying into short positions)
        elif mean_delta > 0 and ev_up > 0.015:
            side = 'Up'
            intensity = min(1.0, abs(funding_delta_conf) * 5000)
            bet_fraction = 0.7 + (0.3 * intensity)

    # --- REGIME 2: STANDARD MOMENTUM (Trend Following) ---
    # If no squeeze is detected, we look for aligned intent and action.
    elif max(ev_up, ev_down) > 0.035:
        # Check for alignment: Delta and OBI should be moving together
        if obi_delta_align > 0 and mean_delta > 0 and ev_up > ev_down:
            side = 'Up'
            # Scale down bet size based on volatility to avoid high-noise environments
            vol_penalty = 1.0 / (1.0 + mean_vol * 5)
            bet_fraction = 0.5 * vol_penalty
            
        elif obi_delta_align < 0 and mean_delta < 0 and ev_down > ev_up:
            side = 'Down'
            vol_penalty = 1.0 / (1.0 + mean_vol * 5)
            bet_fraction = 0.5 * vol_penalty

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0