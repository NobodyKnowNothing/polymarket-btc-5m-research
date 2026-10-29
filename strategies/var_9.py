# Var 9 | BT: 2.695 | Live PnL: $-0.47 | Trades: 1
# Desc: Implements "Absorption-Consensus Adaptive Hurdle," which replaces multiplicative multipliers with a subtractive penalty based on market fragmentation (price/OBI spreads) and absorption (delta-to-price

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features by calculating 
    absorption ratios and normalized market fragmentation metrics.
    """
    # Create a copy to avoid SettingWithCopyWarning and drop metadata
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Absorption Ratio: High delta with low price movement suggests absorption (liquidity walls).
    # We use price_vs_strike_pct to represent the price displacement.
    # Adding a small epsilon to avoid division by zero.
    X['absorption_ratio'] = X['sum_delta'].abs() / (X['price_vs_strike_pct'].abs() + 1e-7)

    # 2. Price Fragmentation: Normalized price spread across exchanges.
    # price_spread is absolute, so we convert to percentage.
    X['price_spread_pct'] = X['price_spread'] / (X['binance_price'] + 1e-7)

    # 3. OBI Fragmentation: Normalized OBI spread across exchanges.
    # Since OBI is [-1, 1], the max possible spread is 2.0.
    X['obi_spread_norm'] = X['obi_spread'] / 2.0

    # Final cleanup: handle NaNs and infinities
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for high regularization to prevent overfitting noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 10.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using an Adaptive Subtractive Hurdle.
    Instead of multiplying EV (which is volatile), we subtract a penalty 
    representing market noise, absorption, and fragmentation.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered penalty features
    # absorption_ratio: how much delta is required to move the price
    # price_spread_pct: how much exchanges disagree on price
    # obi_spread_norm: how much exchanges disagree on order book imbalance
    # mean_volatility: current volatility regime
    abs_ratio = features_row.get('absorption_ratio', 0.0)
    p_spread = features_row.get('price_spread_pct', 0.0)
    o_spread = features_row.get('obi_spread_norm', 0.0)
    vol = features_row.get('mean_volatility', 0.001)

    # 3. Calculate Subtractive Penalty
    # We penalize:
    # - High absorption (logarithmic to prevent extreme outliers from killing all trades)
    # - High price disagreement (linear)
    # - High OBI disagreement (linear)
    penalty = (0.015 * np.log1p(abs_ratio)) + (1.0 * p_spread) + (0.2 * o_spread)

    # 4. Adjusted EV and Execution
    # We only trade if the model's edge significantly exceeds the market noise/fragmentation.
    adj_ev = best_ev - penalty
    hurdle = 0.025

    if adj_ev > hurdle:
        # Time-decay: Reduce exposure as the window closes to mitigate last-second volatility
        time_factor = max(0.3, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Use a volatility-adjusted approach.
        # We want to scale the bet based on the strength of the adjusted edge 
        # relative to the current market volatility.
        # denominator is scaled to keep bet_fraction in a reasonable 0.0-1.0 range.
        bet_fraction = adj_ev / (vol * 50.0 + 0.05)
        bet_fraction = min(1.0, bet_fraction) * time_factor

        # Minimum bet threshold to ensure meaningful trades
        if bet_fraction < 0.05:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0