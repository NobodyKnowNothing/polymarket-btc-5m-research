# Var 143 | BT: 0.574 | Live PnL: $-26.60 | Trades: 60
# Desc: Implements "Volatility-Gated Funding-Exhaustion (VGFE)," which uses the divergence between order book intent (OBI) and aggressive flow (Delta) to identify exhaustion, gated by funding extremes and pri
# Stats: {"live_returns": [-1.0, -0.5131727457046509, 0.04885600000000001, 0.10877132415771484, -1.0, 0.15033358335494995, 1.890848, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, 0.06987699999999997, -1.0, -1.0, -1.0, 0.03413569927215576, 1.5191249999999998, 0.03413569927215576, -1.0, -1.0, 0.03600156307220459, 0.13876599999999994, -1.0, -1.0, 0.11838960647583008, -1.0, 1.233902, -1.0, 0.03600156307220459, -1.0, 0.05925399999999992, 0.11842095851898193, -1.0, 0.04885600000000001, -1.0, -1.0, 0.10316799999999993, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -0.8863531351089478, -1.0, -1.0, -1.0, -1.0, 1.890848, -1.0], "live_pnl": -26.604394912719727, "live_wins": 21, "live_total": 60, "age": 446, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 0.5740071064062924, "backtest_acc": 0.54173197492163, "backtest_n": 631}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the stability of microstructure signals relative to volatility.
    """
    df = df.copy()

    # 1. Microstructure Stability Ratios: How much 'intent' or 'flow' exists per unit of volatility.
    # High values suggest a strong, stable signal; low values suggest noise.
    df['obi_vol_ratio'] = df['mean_obi'] / (df['mean_volatility'] + 1e-6)
    df['delta_vol_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 2. Funding Intensity: Combines the rate and the total volume of funding.
    df['funding_intensity'] = df['mean_funding'] * df['sum_funding']

    # 3. Price Velocity: The speed of price movement relative to the strike.
    df['price_velocity'] = df['price_vs_strike_pct'] / (df['seconds_remaining'] + 1e-6)

    # 4. Price Stretch: The absolute percentage distance from the strike.
    df['price_stretch'] = df['price_vs_strike_pct'].abs()

    # 5. Coherence: The product of OBI and Delta (alignment).
    df['microstructure_coherence'] = df['mean_obi'] * df['mean_delta']

    # Drop metadata and target as per instructions
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Gated Funding-Exhaustion (VGFE)".

    Logic:
    1. Trend Mode: If OBI and Delta are aligned (both positive or both negative), 
       we follow the trend if the EV is sufficient.
    2. Reversal Mode: If OBI and Delta diverge (e.g., aggressive buying but 
       order book is heavy on the sell side), we look for an exhaustion reversal.
       This is ONLY triggered if:
       - Funding is extreme (indicating an overextended position).
       - Price is stretched (price is significantly far from the strike).
    3. Sizing: Scales bet size by the magnitude of the edge, inversely by 
       volatility, and decays as the market window closes.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered and raw features
    mean_obi = features_row.get('mean_obi', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_funding = features_row.get('mean_funding', 0.0)
    price_stretch = features_row.get('price_stretch', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    obi_sign = np.sign(mean_obi)
    delta_sign = np.sign(mean_delta)

    side = None
    best_ev = 0.0

    # --- 1. TREND FOLLOWING REGIME ---
    # OBI and Delta agree (e.g., both buying)
    if obi_sign == delta_sign and obi_sign != 0:
        if obi_sign > 0 and ev_up > 0.03:
            side, best_ev = 'Up', ev_up
        elif obi_sign < 0 and ev_down > 0.03:
            side, best_ev = 'Down', ev_down

    # --- 2. REVERSAL (EXHAUSTION) REGIME ---
    # OBI and Delta disagree (e.g., aggressive buying but no book support)
    elif obi_sign != delta_sign and obi_sign != 0:
        # Gating: Reversals are high-noise; require funding stress and price extension
        is_funding_extreme = abs(mean_funding) > 1e-5
        is_extended = price_stretch > 0.0015 # 0.15% stretch

        if is_funding_extreme and is_extended:
            # If Delta is positive (aggressive buying) but OBI is negative (selling pressure), bet Down
            if delta_sign > 0 and ev_down > 0.04:
                side, best_ev = 'Down', ev_down
            # If Delta is negative (aggressive selling) but OBI is positive (buying pressure), bet Up
            elif delta_sign < 0 and ev_up > 0.04:
                side, best_ev = 'Up', ev_up

    # --- 3. EXECUTION & SIZING ---
    if side:
        # Scale bet fraction by:
        # a) Edge magnitude (target 4% EV for full bet)
        # b) Volatility penalty (lower bet in high vol)
        # c) Time decay (reduce risk in the final 30 seconds)
        
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        vol_penalty = 1.0 / (1.0 + mean_vol)
        
        bet_fraction = (best_ev / 0.04) * vol_penalty * time_factor
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0