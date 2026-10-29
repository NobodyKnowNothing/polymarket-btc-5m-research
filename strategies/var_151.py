# Var 151 | BT: 0.588 | Live PnL: $-5.76 | Trades: 60
# Desc: Implements "Price-Flow Efficiency Gating (PFEG)," which identifies whether price movement is "efficient" (trending with flow) or "inefficient" (absorption or liquidity vacuum) to select the optimal di
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -0.7422249913215637, -1.0, 0.020025968551635742, 2.651597, -1.0, -1.0, 1.5191249999999998, 0.08587318658828735, -1.0, -1.0, 4.893558, -1.0, 2.073133, 1.9791409999999998, 1.9791409999999998, 0.9876765012741089, 2.073133, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, 0.05925399999999992, -1.0, -1.0, -1.0, 0.07802265882492065, -1.0, -1.0, 1.4562249999999999, 4.245841, -1.0, -1.0, -1.0, 2.518758, 1.1846420000000002, -1.0, 0.11476700000000006, -1.0, -0.8171059489250183, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998], "live_pnl": -5.764881134033203, "live_wins": 21, "live_total": 60, "age": 184, "rounds_underwater": 9, "abstain_rounds": 0, "backtest_score": 0.5877448945798195, "backtest_acc": 0.5355097365406644, "backtest_n": 1794}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    efficiency of price movement relative to aggressive order flow.
    """
    df = df.copy()

    # 1. Price-Flow Efficiency: How much price moves per unit of aggressive delta.
    # High efficiency -> "Vacuum/Trend" (Price moves easily with flow).
    # Low efficiency -> "Absorption" (Price resists flow).
    # We use a denominator offset to keep the ratio stable.
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff()
    df['efficiency_ratio'] = df['price_change'] / (df['mean_delta'].abs() + 1.0)

    # 2. Directional Signals
    df['delta_sign'] = np.sign(df['mean_delta'])
    df['price_sign'] = np.sign(df['price_change'])

    # 3. Momentum of Intent
    # Captures if the order book imbalance is accelerating or decelerating.
    df['obi_acceleration'] = df.groupby('market_slug')['mean_obi'].diff()

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for detecting regime-based non-linearities."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Price-Flow Efficiency Gating (PFEG)".

    Logic:
    1. Determine Microstructure Regime:
       - Vacuum/Trend Regime: High efficiency (price moves easily). Follow price direction.
       - Absorption Regime: Low efficiency (price resists delta). Trade against delta.
       - Standard Regime: Moderate efficiency. Follow delta direction.
    2. Validate with Model:
       - Only execute if the model's predicted probability aligns with the regime's target side.
    3. Dynamic Sizing:
       - Scale by edge (EV) and reduce exposure as the market window closes.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    eff = features_row.get('efficiency_ratio', 0.0)
    d_sign = features_row.get('delta_sign', 0.0)
    p_sign = features_row.get('price_sign', 0.0)

    side = None
    
    # Thresholds for regime classification
    # High efficiency (> 0.1) means price is "slippery" (Vacuum)
    # Low efficiency (< 0.05) means price is "sticky" (Absorption)
    if eff > 0.1:
        # --- VACUUM/TREND REGIME ---
        # Price is moving strongly; follow the price direction
        target_side = 'Up' if p_sign > 0 else ('Down' if p_sign < 0 else None)
    elif eff < 0.05 and eff > -0.05:
        # --- ABSORPTION REGIME ---
        # Delta is high but price isn't moving; trade the reversal
        target_side = 'Down' if d_sign > 0 else ('Up' if d_sign < 0 else None)
    else:
        # --- STANDARD REGIME ---
        # Follow the aggressive flow (Delta)
        target_side = 'Up' if d_sign > 0 else ('Down' if d_sign < 0 else None)

    # 1. Execution Logic: Match target side with positive EV
    if target_side == 'Up' and ev_up > 0.025:
        side = 'Up'
        best_ev = ev_up
    elif target_side == 'Down' and ev_down > 0.025:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # 2. Dynamic Bet Sizing
    # Scale by EV (target 4% for full bet)
    # Scale by time (reduce risk in the final 60 seconds)
    time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
    ev_factor = min(1.0, best_ev / 0.04)
    
    bet_fraction = ev_factor * time_factor
    
    return side, min(1.0, max(0.02, bet_fraction))