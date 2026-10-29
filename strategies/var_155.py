# Var 155 | BT: -0.082 | Live PnL: $-11.55 | Trades: 60
# Desc: Implements "Coherence-Exhaustion Gating (CEG)," which scales conviction based on the alignment of Delta, OBI, and Model prediction, while penalizing trades where the price is already overextended rela
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -0.9119774699211121, -1.0, 1.5191249999999998, 0.06877249479293823, 2.651597, -1.0, -1.0, 1.0171962976455688, -0.4571177363395691, -1.0, -0.39568057656288147, -1.0, -1.0, -1.0, 2.073133, -1.0, 0.04126349091529846, 1.4562249999999999, 1.9791409999999998, 0.7508633136749268, 1.9791409999999998, -1.0, 0.9603331685066223, 2.073133, -0.88786381483078, -1.0, -0.5772374272346497, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -0.7618628740310669, -1.0, -1.0, 0.5555557608604431, -1.0, 0.6844501495361328, -1.0, -0.5129948258399963, -1.0, -0.3751662075519562, -1.0, 1.28551, -1.0, -0.634682834148407, -1.0, -1.0, -1.0, 1.890848, 0.04971921443939209, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0], "live_pnl": -11.55138111114502, "live_wins": 19, "live_total": 60, "age": 88, "rounds_underwater": 27, "abstain_rounds": 0, "backtest_score": -0.08196396326851137, "backtest_acc": 0.5630553979958405, "backtest_n": 4288}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price-to-strike positioning and microstructure coherence.
    """
    df = df.copy()

    # 1. Price Position: How far is the price from the strike? 
    # Essential for the model to learn mean-reversion vs momentum.
    df['price_diff_pct'] = (df['binance_price'] - df['strike_price']) / df['strike_price']

    # 2. Microstructure Coherence: Do OBI and Delta point in the same direction?
    # 1.0 = perfect agreement, -1.0 = total disagreement.
    df['delta_obi_sync'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 3. Volatility-Normalized Flow: How much "effort" (delta) is required to move price?
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and raw price columns that don't generalize well
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'binance_price', 'strike_price']
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
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Coherence-Exhaustion Gating (CEG)".

    Logic:
    1. Identify the side with the highest Expected Value (EV).
    2. Coherence Check: If the model's direction is not supported by both 
       Order Book Imbalance (OBI) and Aggressive Flow (Delta), reduce bet size.
    3. Exhaustion Check: If the price is already significantly extended in the 
       predicted direction (e.g., > 0.15%), reduce bet size to avoid "buying the top".
    4. Time Decay: Reduce conviction as the market window closes to avoid late-game noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    if ev_up > ev_down:
        side, best_ev = 'Up', ev_up
        target_sign = 1
    else:
        side, best_ev = 'Down', ev_down
        target_sign = -1

    # Minimum hurdle for any trade
    if best_ev < 0.025:
        return None, 0.0

    # 1. Coherence Component
    # We want Delta and OBI to agree with our chosen side.
    coherence = 1.0
    if np.sign(features_row.get('mean_delta', 0)) != target_sign:
        coherence *= 0.6
    if np.sign(features_row.get('mean_obi', 0)) != target_sign:
        coherence *= 0.6

    # 2. Exhaustion Component
    # If we want to go 'Up' but price is already way above strike, it's overextended.
    exhaustion = 1.0
    price_diff = features_row.get('price_diff_pct', 0.0)
    if side == 'Up' and price_diff > 0.0015:
        exhaustion = 0.4
    elif side == 'Down' and price_diff < -0.0015:
        exhaustion = 0.4

    # 3. Time Component
    # As seconds_remaining approaches 0, we become more cautious.
    time_factor = 1.0
    if seconds_remaining < 120:
        time_factor = max(0.4, seconds_remaining / 120.0)

    # Calculate final bet fraction
    # Base bet is scaled by how much our EV exceeds a 4% target.
    bet_fraction = (best_ev / 0.04) * coherence * exhaustion * time_factor
    
    return side, min(1.0, max(0.0, bet_fraction))