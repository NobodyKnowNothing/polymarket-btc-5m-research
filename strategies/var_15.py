# Var 15 | BT: 1.671 | Live PnL: $-0.70 | Trades: 5
# Desc: Implements "Cross-Exchange Fragmentation & Flow-Coherence Gating," which uses the ratio of price dispersion to volatility as a primary risk-scaling factor and penalizes trades where aggressive delta a
# Stats: {"live_returns": [0.5942406058311462, -0.7483219504356384, -0.09819990396499634, -0.2173490673303604, -0.2344283163547516], "live_pnl": -0.7040586471557617, "live_wins": 1, "live_total": 5, "age": 96, "rounds_underwater": 4, "abstain_rounds": 60, "backtest_score": 1.670636769334291, "backtest_acc": 0.5613800991231415, "backtest_n": 1065}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange dispersion and microstructure intent alignment.
    """
    # Drop metadata but keep pm_costs for model context
    # seconds_remaining is metadata for the filter, not the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Relative Price Dispersion: Measures how "fragmented" the price is across exchanges
    # relative to the current volatility regime. High values suggest high arbitrage risk.
    X['rel_price_spread'] = X['price_spread'] / (X['mean_volatility'] + 1e-7)

    # 2. Flow-Book Alignment: Directional agreement between aggressive flow (Delta) 
    # and passive intent (OBI). 1 = perfect alignment, -1 = divergence, 0 = neutral.
    X['flow_obi_alignment'] = np.sign(X['sum_delta']) * np.sign(X['mean_obi'])

    # 3. Flow Intensity: Magnitude of aggressive flow relative to volatility.
    # Helps distinguish between "breakout" flow and "noise" flow.
    X['delta_vol_ratio'] = X['sum_delta'].abs() / (X['mean_volatility'] + 1e-7)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to combat 5-min noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Fragmentation-Adjusted Expected Value.
    The hurdle scales up when exchanges are out of sync (high dispersion)
    or when flow and book intent are in conflict (absorption/divergence).
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

    # 2. Extract engineered risk/quality metrics
    rel_spread = features_row.get('rel_price_spread', 0)
    alignment = features_row.get('flow_obi_alignment', 0)

    # 3. Calculate Dynamic Hurdle
    # Base hurdle is 2.5%.
    # We add a penalty for price fragmentation (rel_spread).
    # We add a penalty if flow and OBI are not aligned (alignment <= 0).
    fragmentation_penalty = 0.05 * min(1.0, rel_spread)
    alignment_penalty = 0.03 if alignment <= 0 else 0.0
    
    dynamic_hurdle = 0.025 + fragmentation_penalty + alignment_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: Reduce exposure as the window closes to avoid resolution noise.
        time_factor = max(0.5, seconds_remaining / 300.0)

        # Bet sizing: Scale by the strength of the edge relative to the hurdle,
        # but aggressively scale down if the market is highly fragmented.
        # (1 / (1 + rel_spread)) ensures large spreads lead to much smaller bets.
        bet_fraction = (best_ev / dynamic_hurdle) * (1.0 / (1.0 + rel_spread)) * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold for meaningful trades
        if bet_fraction < 0.05:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0