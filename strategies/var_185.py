# Var 185 | BT: -0.076 | Live PnL: $-0.59 | Trades: 112
# Desc: Implements "Liquidity-Sweep & Proximity-Gated Hybrid (LSPG)," which targets high-conviction momentum during "sweeps" (where aggressive delta opposes passive OBI) specifically when the price is near th
# Stats: {"live_returns": [-1.0, -1.0, 1.3964720000000002, -0.9623649716377258, 0.10870939493179321, -1.0, -1.0, -1.0, -1.0, 0.49914461374282837, -1.0, -1.0, -1.0, 1.5854279999999998, 4.245841, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, 1.1375739999999999, -1.0, 0.12663000000000002, 1.1846420000000002, 1.9791409999999998, 2.518758, -1.0, 1.5854279999999998, -1.0, -0.991019606590271, 1.8077519999999998, 1.3964720000000002, 2.950496, 1.3396370000000002, -0.5326433181762695, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, 2.173394, -1.0, 2.950496, -1.0, -1.0, -1.0, 2.280573, -1.0, -1.0, -1.0, 0.09182599999999996, -0.8119054436683655, 3.1194439999999997, -1.0, 0.035984039306640625, -1.0, -1.0, 0.10316799999999993, 1.9791409999999998, 0.12663000000000002, 1.890848, 0.08073200000000003, -1.0, 1.5191249999999998, 1.049452, 1.233902, -1.0, 1.092555, 0.045812785625457764, -1.0, -1.0, 0.12663000000000002, 1.890848, -1.0, 0.08073200000000003, 2.651597, -1.0, -1.0, -1.0, 0.11476700000000006, 0.06987699999999997, 0.09182599999999996, 0.16389999999999993, -0.7744764685630798, -1.0, -0.40839189291000366, 0.08073200000000003, -1.0, -0.9011437892913818, 0.056479811668395996, 0.06047332286834717, 2.39541, -1.0, 0.03599095344543457, -0.9307125210762024, 0.03598678112030029, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, 2.173394, -1.0, -1.0, -1.0, 0.09182599999999996, 1.8077519999999998, 0.12663000000000002, -0.9305022358894348, -1.0, -1.0], "live_pnl": -0.5941603183746338, "live_wins": 52, "live_total": 112, "age": 189, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": -0.0760570880578117, "backtest_acc": 0.5638183768323204, "backtest_n": 3572}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the interaction between aggressive flow (Delta) and passive intent (OBI)
    near the strike price.
    """
    df = df.copy()

    # 1. Delta-OBI Mismatch: Detects "Sweeps"
    # -1 if Delta and OBI have opposite signs (Aggressive flow is clearing the book)
    #  1 if Delta and OBI have the same sign (Consensus)
    df['delta_obi_mismatch'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 2. Strike Proximity: How close is the current price to the resolution strike?
    # High value (near 1.0) means we are very close to the strike.
    df['strike_proximity'] = 1.0 / (1.0 + df['price_vs_strike_pct'].abs())

    # 3. Flow-to-Volatility Ratio: Measures the intensity of the flow relative to noise.
    # High value means aggressive flow is significant compared to recent volatility.
    df['flow_vol_ratio'] = df['mean_delta'].abs() / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = [c for c in ['market_slug', 'timestamp', 'actual_outcome'] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Final cleanup: Handle potential division by zero or NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters with moderate regularization."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using the "Liquidity-Sweep & Proximity-Gated Hybrid (LSPG)" logic.

    Logic:
    1. Identify "Sweep" Regime: 
       - Occurs when delta_obi_mismatch < 0 (Aggressive flow opposes passive intent)
       - AND price is near the strike (strike_proximity > 0.7)
       - AND flow is intense (flow_vol_ratio > 0.2)
    2. Sweep Execution:
       - If in Sweep regime, follow the direction of the aggressive Delta.
       - Use a lower EV hurdle (0.015) because sweeps are high-velocity.
    3. Model Execution (Fallback):
       - If not in a sweep, rely on the XGBoost model's predicted edge.
       - Use a higher EV hurdle (0.035) to avoid noise.
    4. Dynamic Sizing:
       - Scale bet size by the magnitude of the edge and the proximity to the strike.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    mismatch = features_row.get('delta_obi_mismatch', 0.0) < 0
    proximity = features_row.get('strike_proximity', 0.0)
    flow_vol = features_row.get('flow_vol_ratio', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)

    side = None
    best_ev = 0.0

    # 1. Regime Detection & Decision
    if mismatch and proximity > 0.7 and flow_vol > 0.2:
        # --- SWEEP REGIME ---
        # The aggressive flow is likely sweeping through the opposing passive orders.
        if mean_delta > 0 and ev_up > 0.015:
            side = 'Up'
            best_ev = ev_up
        elif mean_delta < 0 and ev_down > 0.015:
            side = 'Down'
            best_ev = ev_down
    else:
        # --- STANDARD REGIME ---
        # Rely on the model's calibrated probability.
        if ev_up > 0.035:
            side = 'Up'
            best_ev = ev_up
        elif ev_down > 0.035:
            side = 'Down'
            best_ev = ev_down

    # 2. Execution and Dynamic Sizing
    if side:
        # Scale bet fraction by:
        # a) Edge magnitude (target 4% EV for full bet)
        # b) Proximity (higher near strike, where impact is greatest)
        
        # Ensure proximity-based scaling doesn't drop too low
        proximity_factor = 0.5 + (0.5 * proximity)
        
        # Time decay: reduce size in the final 30 seconds to avoid whipsaws
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = (best_ev / 0.04) * proximity_factor * time_factor

        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0