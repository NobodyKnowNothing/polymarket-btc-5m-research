# Var 200 | BT: -0.035 | Live PnL: $-9.60 | Trades: 60
# Desc: Targets "Order-Flow Traps" (OFT) where aggressive market orders (Delta) are being absorbed by opposing passive liquidity (OBI) at price extremes, signaling a high-probability mean-reversion.
# Stats: {"live_returns": [1.3271009469502577, 0.9676930743129868, 0.6917628000000001, -0.8330695466666665, -0.2755825638095238, -0.184074408, -0.4, -0.4, -0.4, -0.4, -0.4, 0.4368684816822043, -0.4, 0.6018250255240194, -0.4, 0.5980774870000001, -1.0, -0.27272727272727276, -0.39301569714285717, -0.4, 0.8582146728613715, -0.4, 0.05054972864467555, -0.57251264, -0.4, -1.0, 0.9581640000000001, -0.4, -0.7311422222222221, -0.4, -0.4, -1.0, -1.0, -0.4, 2.173394, -0.2932118000000001, 0.58249, -0.1344120763636364, -0.4, -0.15895716000000004, -0.6125842666666667, 0.5358548000000001, -0.4, -0.5324310044444445, -0.2, -0.2, 0.58249, 0.09182599999999996, -1.0, 0.028705999999999898, 0.07076719999999997, -0.42084207428571446, -0.4, -0.4, 0.7094640000000001, 0.02031046191893332, -0.4735376266666666, -1.0, -0.4, -1.0], "live_pnl": -9.602541680101224, "live_wins": 18, "live_total": 60, "age": 176, "rounds_underwater": 44, "abstain_rounds": 0, "backtest_score": -0.03460039821702291, "backtest_acc": 0.5643845335225257, "backtest_n": 1998}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between aggressive flow and passive intent (The Trap).
    """
    df = df.copy()

    # 1. Delta-OBI Divergence (The "Trap" Signal)
    # Negative: Aggressive buying (Delta > 0) vs Negative OBI (Sell walls present)
    # Positive: Aggressive selling (Delta < 0) vs Positive OBI (Buy walls present)
    df['delta_obi_divergence'] = df['mean_delta'] * df['mean_obi']

    # 2. Delta Efficiency (How much delta is required to move the price)
    # Low efficiency suggests high absorption/resistance.
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 3. Delta Velocity (Acceleration of aggressive flow)
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 4. Volatility-Normalized Delta
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

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
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Order-Flow Trap (OFT)" logic.

    Logic:
    1. Identify a Trap:
       - A trap occurs when `delta_obi_divergence` is extreme.
       - If divergence is strongly negative: Aggressive buyers are hitting a massive sell wall (OBI is negative).
       - If divergence is strongly positive: Aggressive sellers are hitting a massive buy wall (OBI is positive).
    2. Contextualize with Price:
       - Traps are most explosive when the price is already extended from the strike.
    3. Execution:
       - If a trap is detected AND the model's EV aligns with the reversal, bet heavily.
    """
    # Exit if too little time remains to allow for the move to resolve
    if seconds_remaining < 45:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    div = features_row.get('delta_obi_divergence', 0.0)
    price_ext = abs(features_row.get('price_vs_strike_pct', 0.0))
    mean_vol = features_row.get('mean_volatility', 0.0)

    side = None
    bet_fraction = 0.0

    # Thresholds for "Trap" detection
    # We use a threshold that scales with the intensity of the divergence
    trap_threshold = 0.08 
    
    # Minimum price extension to ensure we aren't just trading noise in the middle
    min_extension = 0.0001 

    # --- REGIME: THE TRAP (Mean Reversion) ---
    if abs(div) > trap_threshold and price_ext > min_extension:
        
        # Case A: Long Trap (Buyers trapped by sell walls) -> Bet DOWN
        if div < 0 and ev_down > 0.02:
            side = 'Down'
            # Scale bet by the magnitude of the trap and the model's conviction
            # We scale down slightly in extremely high volatility to avoid whipsaws
            vol_adj = 1.0 / (1.0 + mean_vol * 2)
            intensity = min(1.0, abs(div) * 4.0)
            bet_fraction = intensity * vol_adj

        # Case B: Short Trap (Sellers trapped by buy walls) -> Bet UP
        elif div > 0 and ev_up > 0.02:
            side = 'Up'
            vol_adj = 1.0 / (1.0 + mean_vol * 2)
            intensity = min(1.0, abs(div) * 4.0)
            bet_fraction = intensity * vol_adj

    # --- REGIME: STANDARD TREND (Momentum) ---
    # If no trap is detected, fall back to standard EV-based momentum if edge is high
    elif max(ev_up, ev_down) > 0.04:
        # Only take momentum if it's not "exhausted" (delta and OBI aligned)
        # If they are aligned, it's a healthy trend.
        if ev_up > ev_down and features_row.get('mean_obi', 0) > 0:
            side = 'Up'
            bet_fraction = 0.4
        elif ev_down > ev_up and features_row.get('mean_obi', 0) < 0:
            side = 'Down'
            bet_fraction = 0.4

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0