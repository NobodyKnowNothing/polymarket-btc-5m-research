# Var 70 | BT: 0.086 | Live PnL: $-4.13 | Trades: 61
# Desc: Implements "Microstructure-Regime Divergence Gating (MRDG)," which uses a discrete regime switch between high and low volatility to fundamentally change the trading logic: in high volatility, it ampli
# Stats: {"live_returns": [0.183442, -0.7, -1.0, -1.0, -0.5341981649398804, 0.05651240000000002, 0.08033690000000004, 0.8637314, 2.950496, 0.6513514, 0.09182599999999996, 2.4539375000000003, 4.008452699999999, 0.05351740121841431, 0.3822791000000001, -0.7, 0.7346164, 2.3126292, -0.7, -0.7, 0.3062283000000001, 0.5775230999999998, -0.7, -1.0, -1.0, -0.44407883286476135, 0.3594226837158203, -1.0, 0.27885689999999996, -0.7, -0.7, -0.7, 0.8998569999999999, -0.7, -0.7, -1.0, -1.0, 0.15253490000000003, 0.08073200000000003, -1.0, -1.0, -1.0, -0.595922589302063, -1.0, -1.0, -0.7, 0.3062283000000001, 3.1854522000000003, 0.027072500000000055, -0.7, -1.0, -1.0, 0.5327846999999999, -0.19608508050441742, 1.28551, -0.41454043984413147, -0.8599154949188232, 1.0193575, -0.8415455222129822, -0.7, -0.974564254283905], "live_pnl": -4.126159191131592, "live_wins": 26, "live_total": 61, "age": 110, "rounds_underwater": 20, "abstain_rounds": 0, "backtest_score": 0.08598407452704743, "backtest_acc": 0.5384006334125099, "backtest_n": 2242}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement and the absorption of aggressive flow.
    """
    df = df.copy()

    # 1. Price Velocity: Recent change in the primary exchange price
    # Grouping by market_slug to prevent cross-market leakage
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Absorption Ratio: How much aggressive flow (Delta) is required to move the price.
    # High values indicate massive absorption (aggressive orders hitting limit orders without moving price).
    # This is a key precursor to mean-reversion/squeezes.
    df['absorption_ratio'] = (df['mean_delta'].abs() / (df['price_change'].abs() + 1e-6)).fillna(0)

    # 3. Delta Efficiency: The price movement generated per unit of delta.
    # Positive = price moves with delta (trending); Negative = price moves against delta (absorption/reversal).
    df['delta_efficiency'] = (df['price_change'] / (df['mean_delta'].abs() + 1e-6)).fillna(0)
    # Clip to prevent extreme outliers from destabilizing XGBoost
    df['delta_efficiency'] = df['delta_efficiency'].clip(-10, 10)

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
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Regime Divergence Gating (MRDG)".

    Logic:
    The strategy splits the market into two distinct behavioral regimes:
    
    1. HIGH VOLATILITY (Trend Following):
       We trust the model's direction but scale the bet based on 'delta_efficiency'.
       If the model says 'Up' and the delta is efficiently moving the price up, we bet full.
       If the model says 'Up' but the delta is being absorbed (inefficient), we drastically reduce the bet.

    2. LOW VOLATILITY (Microstructure Reversal):
       The model is often noise in low-vol ranges. Instead, we look for 'Absorption Events'.
       If we see high aggressive delta (mean_delta) but almost no price movement (high absorption_ratio),
       we execute a high-conviction reversal trade, overriding the model's direction.
    """
    # 1. Basic EV Calculation
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    best_ev = max(ev_up, ev_down)
    side = 'Up' if ev_up > ev_down else 'Down'

    # 2. Time-gate
    if seconds_remaining < 30:
        return None, 0.0

    # 3. Extract engineered features
    # Note: we use .get() to handle potential missing columns during inference
    abs_ratio = features_row.get('absorption_ratio', 0.0)
    delta_eff = features_row.get('delta_efficiency', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    
    # Handle volatility regime (can be string 'high'/'low' or a float)
    vol_val = features_row.get('mean_volatility', 0.0)
    is_high_vol = (vol_val == 'high') or (isinstance(vol_val, (int, float)) and vol_val > 0.0005)

    # 4. Regime-Based Decision Logic
    if is_high_vol:
        # --- HIGH VOLATILITY REGIME: Momentum/Trend Validation ---
        if best_ev < 0.025:
            return None, 0.0
        
        # If price movement is inefficient (delta_eff < 0), the trend is being absorbed.
        # We reduce bet size to protect against a reversal.
        multiplier = 1.0 if delta_eff > 0 else 0.25
        bet_fraction = (best_ev / 0.04) * multiplier
        
        return side, max(0.0, min(1.0, bet_fraction))

    else:
        # --- LOW VOLATILITY REGIME: Absorption Reversal ---
        # Look for high-conviction absorption: aggressive flow with no price movement.
        if abs(mean_delta) > 1e-4 and abs_ratio > 25.0:
            # If delta is positive (buying) but price isn't moving, bet 'Down' (reversal)
            # If delta is negative (selling) but price isn't moving, bet 'Up' (reversal)
            rev_side = 'Down' if mean_delta > 0 else 'Up'
            return rev_side, 0.7  # High conviction reversal bet
        
        # In low vol, if no absorption is detected, we abstain to avoid noise.
        return None, 0.0