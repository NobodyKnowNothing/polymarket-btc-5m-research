# Var 188 | BT: 1.767 | Live PnL: $0.00 | Trades: 0
# Desc: Implements "Regime-Adaptive Signal Switching (RASS)," which switches the primary predictive signal between Aggressive Flow (Delta) in high-volatility momentum regimes and Passive Intent (OBI) in low-v
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 59, "rounds_underwater": 0, "abstain_rounds": 59, "backtest_score": 1.7665654082766014, "backtest_acc": 0.5567673878061863, "backtest_n": 5}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, introducing 
    regime-specific ratios and temporal price velocity.
    """
    df = df.copy()

    # 1. Temporal Price Dynamics (Grouped by market to prevent leakage)
    # Captures the immediate velocity of price movement within the 5-min window
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['price_change_pct'] = df.groupby('market_slug')['binance_price'].pct_change().fillna(0)

    # 2. Regime-Specific Ratios
    # High Volatility -> Momentum (Delta matters more)
    # Low Volatility -> Absorption (OBI matters more)
    df['vol_delta_ratio'] = df['mean_delta'].abs() / (df['mean_volatility'] + 0.0001)
    df['vol_obi_ratio'] = df['mean_obi'].abs() / (df['mean_volatility'] + 0.0001)
    
    # 3. Disagreement Index
    # Measures how much the exchange price spread deviates from the current volatility
    df['spread_vol_ratio'] = df['price_spread'] / (df['mean_volatility'] + 0.0001)

    # 4. Flow-Intent Congruence
    # How well the aggressive flow aligns with the passive book intent
    df['delta_obi_congruence'] = df['mean_delta'] * df['mean_obi']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for regime-switching signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.007,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Adaptive Signal Switching (RASS)".

    Logic:
    1. Disagreement Filter: If price_spread is high relative to volatility, 
       the market is fragmented; abstain to avoid noise.
    2. Regime Switching:
       - High Vol Regime: The market is driven by aggressive sweeps. 
         Follow the sign of 'mean_delta'.
       - Low Vol Regime: The market is driven by order book walls. 
         Follow the sign of 'mean_obi'.
    3. Validation: The selected regime signal must align with the ML model's 
       predicted probability.
    4. Dynamic Sizing: Scale by EV, agreement (inverse spread), and time decay.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract features
    mean_vol = features_row.get('mean_volatility', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    price_spread = features_row.get('price_spread', 0.0)
    spread_vol_ratio = features_row.get('spread_vol_ratio', 0.0)

    # --- 1. DISAGREEMENT FILTER ---
    # If the spread between exchanges is too high relative to volatility, 
    # we are in a "fragmented" state where direction is unreliable.
    if spread_vol_ratio > 2.5:
        return None, 0.0

    # --- 2. REGIME SWITCHING ---
    # Threshold for "High Volatility" is heuristic; 
    # in 5-min BTC markets, 0.0005 is a common regime boundary.
    is_high_vol = mean_vol > 0.0005
    
    side = None
    best_ev = 0.0

    if is_high_vol:
        # Momentum Regime: Follow the aggressive flow
        if mean_delta > 0:
            side, best_ev = 'Up', ev_up
        elif mean_delta < 0:
            side, best_ev = 'Down', ev_down
    else:
        # Absorption Regime: Follow the passive intent
        if mean_obi > 0:
            side, best_ev = 'Up', ev_up
        elif mean_obi < 0:
            side, best_ev = 'Down', ev_down

    # --- 3. VALIDATION & EXECUTION ---
    if side:
        # Ensure the ML model's probability agrees with the regime signal
        # and that there is a minimum edge threshold.
        if side == 'Up' and (prob_up > 0.52 and ev_up > 0.03):
            pass
        elif side == 'Down' and (prob_down > 0.52 and ev_down > 0.03):
            pass
        else:
            return None, 0.0

        # --- 4. DYNAMIC SIZING ---
        # 1. Edge Scale: Target 4% EV for full bet
        edge_scale = best_ev / 0.04
        
        # 2. Agreement Multiplier: Reduce size if exchanges disagree (spread is high)
        agreement_mult = 1.0 / (1.0 + price_spread)
        
        # 3. Time Decay: Reduce size in final 60s to mitigate end-of-window noise
        time_mult = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)

        bet_fraction = edge_scale * agreement_mult * time_mult

        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0