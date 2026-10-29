# Var 163 | BT: -0.384 | Live PnL: $-0.56 | Trades: 60
# Desc: Implements "Funding-Fuelled Momentum (FFM)," which scales conviction by identifying "squeeze" regimes where aggressive flow (Delta) moves against the prevailing funding bias.
# Stats: {"live_returns": [1.4562249999999999, 1.9791409999999998, 1.233902, 1.9791409999999998, -1.0, 2.073133, 0.8939649999999999, 1.1375739999999999, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, 1.0081479999999998, -1.0, -1.0, -1.0, -1.0, 0.1445239782333374, -1.0, 1.3964720000000002, -1.0, -0.5178216695785522, -1.0, -1.0, -1.0, 0.11349904537200928, 1.28551, -1.0, 0.15118699999999996, -0.8671649098396301, -1.0, -1.0, 1.890848, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, 0.9943159818649292, -1.0, -1.0, 1.4562249999999999, 0.02477395534515381, 4.245841, -0.9021602869033813, 2.518758, -1.0, -1.0, -1.0, 2.518758, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -0.5586004257202148, "live_wins": 24, "live_total": 60, "age": 88, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": -0.38363397314195324, "backtest_acc": 0.5638098802395209, "backtest_n": 4695}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the interaction between aggressive flow, order book intent, and 
    the 'cost of carry' signaled by funding rates.
    """
    df = df.copy()

    # 1. OBI-Delta Sync: Do the bids/asks align with the aggressive trades?
    df['obi_delta_sync'] = df['mean_obi'] * df['mean_delta']

    # 2. Funding-Flow Mismatch: Is the flow fighting the funding? 
    # (e.g., Buying while funding is negative = Short Squeeze potential)
    df['funding_flow_mismatch'] = df['mean_delta'] * df['mean_funding']

    # 3. Volatility-Adjusted Delta: Magnitude of flow relative to current noise.
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price-to-Strike Momentum: Normalized velocity of price movement.
    df['price_velocity'] = df['price_vs_strike_pct'] / (df['seconds_remaining'] + 1e-6)

    # Drop metadata and target
    # We keep pm_cost_up/down as they are useful for the model to learn market pricing
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
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
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Fuelled Momentum (FFM)".

    Logic:
    1. Identify High-Conviction Regimes:
       - SQUEEZE: Delta and Funding have opposite signs (e.g., buying into negative funding).
         This is a high-probability continuation signal.
       - IMPULSE: Delta and OBI have the same sign.
         This is a standard trend-following signal.
       - ABSORPTION/NEUTRAL: Otherwise, we treat the signal with high skepticism.
    2. Conviction Scaling:
       - Squeeze = 1.5x conviction multiplier.
       - Impulse = 1.0x conviction multiplier.
       - Neutral/Absorption = 0.5x conviction multiplier.
    3. Execution:
       - Apply the conviction multiplier to the Expected Value (EV).
       - Scale bet size by both the edge and the conviction.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine best side
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    mean_funding = features_row.get('mean_funding', 0.0)

    # Determine signs
    delta_sign = np.sign(mean_delta)
    obi_sign = np.sign(mean_obi)
    funding_sign = np.sign(mean_funding)

    # 1. Regime Identification
    conviction = 0.5  # Default: Neutral/Absorption
    
    # Check for Squeeze (Delta and Funding are fighting)
    # We need non-zero signs to avoid false positives on flat markets
    if delta_sign != 0 and funding_sign != 0 and delta_sign != funding_sign:
        conviction = 1.5
    # Check for Impulse (Delta and OBI are aligned)
    elif delta_sign != 0 and obi_sign != 0 and delta_sign == obi_sign:
        conviction = 1.0

    # 2. Adjusted EV Thresholding
    # We use a lower base hurdle because conviction scales it up
    # If conviction is 1.5, effective_ev = 0.02 * 1.5 = 0.03
    # If conviction is 0.5, effective_ev = 0.02 * 0.5 = 0.01
    effective_ev = best_ev * conviction
    
    if effective_ev > 0.02:
        # 3. Dynamic Bet Sizing
        # Scale by: (Edge / Target_Edge) * Conviction * Time_Safety
        # Target edge of 4% (0.04) for full sizing
        time_factor = 1.0 if seconds_remaining > 45 else (seconds_remaining / 45.0)
        
        bet_fraction = (best_ev / 0.04) * conviction * time_factor
        
        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0