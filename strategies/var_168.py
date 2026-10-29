# Var 168 | BT: 1.243 | Live PnL: $-1.39 | Trades: 2
# Desc: Implements "Flow-Price Slippage Gating (FPSG)," which scales conviction by the ratio of aggressive flow to price movement, effectively avoiding "absorption" regimes where high delta fails to move the 
# Stats: {"live_returns": [-0.7426538467407227, -0.6486625075340271], "live_pnl": -1.3913164138793945, "live_wins": 0, "live_total": 2, "age": 118, "rounds_underwater": 2, "abstain_rounds": 59, "backtest_score": 1.24319297144506, "backtest_acc": 0.5421150278293135, "backtest_n": 140}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    'Microstructure Efficiency'—the relationship between aggressive 
    order flow (Delta) and the resulting price movement (Velocity).
    """
    df = df.copy()

    # 1. Price Velocity: How fast is the price moving relative to the remaining time?
    # We clip seconds_remaining to avoid division by zero/extreme spikes at the very end.
    df['price_velocity'] = df['price_vs_strike_pct'] / (df['seconds_remaining'].clip(lower=1) + 1e-6)

    # 2. Delta Efficiency: How much price movement is generated per unit of Delta?
    # High efficiency = "Slippage" (easy to move price). Low efficiency = "Absorption".
    df['delta_efficiency'] = df['price_velocity'] / (abs(df['mean_delta']) + 1e-6)

    # 3. OBI Efficiency: How much price movement is generated per unit of OBI?
    df['obi_efficiency'] = df['price_velocity'] / (abs(df['mean_obi']) + 1e-6)

    # 4. Absorption Index: The inverse of efficiency. 
    # High value means massive delta is being 'absorbed' by the book without price moving.
    df['absorption_index'] = abs(df['mean_delta']) / (abs(df['price_velocity']) + 1e-6)

    # 5. Spread-to-Volatility Ratio: Detects liquidity holes.
    df['spread_vol_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
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
    Decide whether to trade using "Flow-Price Slippage Gating (FPSG)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Calculate a 'Slippage Multiplier' based on the Absorption Index.
       - If the Absorption Index is high (Delta is high but price isn't moving), 
         the multiplier is low. This protects against 'absorption' regimes where 
         momentum trends stall.
       - If the Absorption Index is low (Slippage), the multiplier is near 1.0.
    3. Apply the multiplier to the EV to create an 'Effective EV'.
    4. Only execute if Effective EV exceeds a threshold, and scale bet size 
       by both the raw edge and the slippage multiplier.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    absorption_index = features_row.get('absorption_index', 0.0)
    
    # 1. Calculate Slippage Multiplier
    # We use log1p to dampen the impact of extreme absorption values.
    # If absorption is 0, multiplier is 1.0. As absorption grows, multiplier decays.
    slippage_multiplier = 1.0 / (1.0 + math.log1p(abs(absorption_index)))

    # 2. Adjusted EV Thresholding
    # We require a higher raw EV if the market is showing signs of absorption.
    effective_ev = best_ev * slippage_multiplier

    # Threshold is slightly higher than baseline to ensure quality
    if effective_ev > 0.025:
        # 3. Dynamic Bet Sizing
        # Scale by: (Edge / Target_Edge) * Slippage_Multiplier * Time_Safety
        # Target edge of 4% (0.04) for full sizing.
        time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        bet_fraction = (best_ev / 0.04) * slippage_multiplier * time_factor

        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0