# Var 13 | BT: 2.151 | Live PnL: $-14.87 | Trades: 25
# Desc: Implements "Volatility-Normalized Flow-Book Confluence," which uses the alignment between Binance's aggressive delta and its order book imbalance to dynamically adjust the required EV hurdle and scale

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Binance-led flow-book alignment and volatility-normalized aggression.
    """
    # Drop metadata but keep pm_costs for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Binance Flow-Book Confluence: Alignment between aggressive delta and OBI on the lead exchange
    # High value indicates aggressive buying/selling is backed by order book depth.
    X['binance_flow_confluence'] = np.sign(X['binance_delta']) * X['binance_obi']

    # 2. Volatility-Normalized Delta: Measures the strength of the flow relative to current market noise.
    # This helps the model distinguish between high-conviction moves and high-volatility noise.
    X['vol_norm_delta'] = X['binance_delta'] / (X['mean_volatility'] + 1e-7)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for high-regularization on microstructure signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 12.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Volatility-Normalized Flow-Book Confluence.
    The hurdle increases when the order flow and order book are misaligned, 
    and bet size is scaled by the square root of the confluence to reward high-conviction signals.
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

    # 2. Extract engineered confluence metric
    # confluence is in [-1, 1]. We care about the magnitude of alignment.
    confluence = features_row.get('binance_flow_confluence', 0.0)
    abs_confluence = abs(confluence)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 2%. If confluence is 0 (no alignment), hurdle rises to 5%.
    dynamic_hurdle = 0.02 + (0.03 * (1.0 - abs_confluence))

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: Reduce bet size slightly as we approach the 5-min resolution.
        time_factor = max(0.5, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale linearly with edge relative to hurdle, 
        # but modulated by the square root of confluence to prevent over-leveraging on noise.
        bet_fraction = (best_ev / dynamic_hurdle) * (abs_confluence ** 0.5)
        bet_fraction = min(1.0, bet_fraction * time_factor)

        # Minimum threshold to ensure meaningful trades
        if bet_fraction < 0.05:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0