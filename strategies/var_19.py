# Var 19 | BT: 0.286 | Live PnL: $-13.35 | Trades: 61
# Desc: Implements "Microstructure-Polymarket Divergence (MPD) Capture," which identifies high-conviction trades where aggressive order flow and book pressure significantly deviate from current Polymarket pri
# Stats: {"live_returns": [0.010286152362823486, 3.1194439999999997, -1.0, 0.027252495288848877, -1.0, -1.0, 0.010286152362823486, -0.802068829536438, -1.0, 2.39541, 0.02367579936981201, 1.28551, -1.0, -1.0, 0.8938688635826111, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.027252495288848877, -1.0, -1.0, 0.027252495288848877, 0.027252495288848877, -1.0, 2.073133, 0.027252495288848877, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, 0.03259912133216858, -0.2793954908847809, 1.7294070000000001, -1.0, -1.0, 0.06317371129989624, -1.0, 0.013920873403549194, -1.0, -1.0, -1.0, 0.04951193928718567, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0], "live_pnl": -13.349642753601074, "live_wins": 23, "live_total": 61, "age": 84, "rounds_underwater": 46, "abstain_rounds": 0, "backtest_score": 0.2859597215220318, "backtest_acc": 0.5669362084456424, "backtest_n": 3545}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between exchange microstructure and Polymarket pricing.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Flow Signal: A bounded measure of directional pressure 
    # (Delta * OBI) normalized by price volatility to capture "true" momentum.
    # We use tanh to bound this between -1 and 1 for stability.
    X['flow_signal'] = np.tanh(
        (X['mean_delta'] * X['mean_obi']) / (X['price_std'] + 1e-7)
    )

    # 2. Polymarket Signal: The implied probability from the current market price.
    # Since pm_cost_up is the price of an 'Up' token, it is the market's P(Up).
    # We center it around 0.5.
    X['pm_signal'] = X['pm_cost_up'] - 0.5

    # 3. Divergence: The gap between exchange-driven pressure and PM pricing.
    # A high positive divergence means flow is bullish but PM price is lagging.
    # A high negative divergence means flow is bearish but PM price is lagging.
    X['divergence'] = X['flow_signal'] - X['pm_signal']

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Microstructure-Polymarket Divergence (MPD).
    We scale conviction based on whether the model's predicted direction 
    is supported by a divergence between exchange flow and PM price.
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

    # 2. Extract divergence metric
    divergence = features_row.get('divergence', 0.0)

    # 3. Determine Alignment
    # We are "aligned" if the model wants to go Up and divergence is positive,
    # or if the model wants to go Down and divergence is negative.
    is_aligned = (side == 'Up' and divergence > 0) or (side == 'Down' and divergence < 0)

    # 4. Dynamic Hurdle and Sizing
    # Base hurdle is 3.5%.
    base_hurdle = 0.035
    
    # If we are aligned with the divergence, we increase our conviction (multiplier).
    # If we are trading against the divergence (price is already efficient), we reduce conviction.
    if is_aligned:
        # Scale multiplier up to 2.0 based on divergence magnitude
        conviction_multiplier = 1.0 + min(1.0, abs(divergence) * 2.0)
    else:
        # Reduce conviction to 0.5 if the signal is not supported by the divergence
        conviction_multiplier = 0.5

    # 5. Execution Logic
    if best_ev > base_hurdle:
        # Time-decay: reduce bet size as the window closes to avoid noise
        time_factor = max(0.5, seconds_remaining / 300.0)

        # Bet sizing: Scale by edge, conviction, and time
        # We use the conviction_multiplier to boost/dampen the effective edge
        bet_fraction = (best_ev / base_hurdle) * conviction_multiplier * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure we don't trade tiny, insignificant edges
        if bet_fraction < 0.15:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0