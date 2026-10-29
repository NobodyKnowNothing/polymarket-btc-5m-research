# Var 22 | BT: 0.689 | Live PnL: $-21.27 | Trades: 61
# Desc: Implements "Congruent Flow & Funding-Squeeze Capture (CFSC)," which uses exchange-level OBI-Delta alignment and a funding-divergence multiplier to identify high-conviction liquidation squeezes.
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0933409999999999, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, 1.890848, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, 0.038675000000000015, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, 0.038675000000000015, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.05925399999999992, 1.28551, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -0.778526246547699, -1.0, 0.15118699999999996, 0.15118699999999996, -1.0], "live_pnl": -21.274995803833008, "live_wins": 16, "live_total": 61, "age": 80, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": 0.6893651003511614, "backtest_acc": 0.5579649708090075, "backtest_n": 1804}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    exchange-level congruence and funding-driven squeeze signals.
    """
    # Drop metadata but keep pm_costs for model context
    # seconds_remaining is used in strategy_filter but we drop it from X to follow constraints
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Exchange Congruence: Do OBI and Delta agree on each exchange?
    # For each exchange, if sign(OBI) == sign(Delta), score is 1. If they differ, -1.
    # Summing across 3 exchanges gives a range of [-3, 3].
    X['congruence'] = (
        np.sign(X['binance_obi']) * np.sign(X['binance_delta']) +
        np.sign(X['coinbase_obi']) * np.sign(X['coinbase_delta']) +
        np.sign(X['kraken_obi']) * np.sign(X['kraken_delta'])
    )

    # 2. Microstructure Absorption: Is the aggressive flow being eaten by the book?
    # A negative value indicates "absorption" (e.g., high bids being hit by aggressive sellers).
    X['absorption'] = X['mean_obi'] * X['mean_delta']

    # 3. Squeeze Signal: Captures divergence between funding sentiment and aggressive flow.
    # If funding is positive (longs paying) and delta is negative (aggressive selling), 
    # the product is negative, signaling a potential long liquidation squeeze.
    X['funding_squeeze_signal'] = X['mean_funding'] * X['mean_delta']

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
        'reg_alpha': 30.0,
        'reg_lambda': 45.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using a Congruent-Flow hurdle and a Squeeze-Multiplier.
    The hurdle increases during exchange disagreement or absorption.
    Bet size is amplified by exchange alignment and funding-delta divergence.
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

    # 2. Extract engineered quality metrics
    congruence = features_row.get('congruence', 0.0)
    absorption = features_row.get('absorption', 0.0)
    squeeze_signal = features_row.get('funding_squeeze_signal', 0.0)

    # 3. Dynamic Hurdle
    # Base hurdle is 3%.
    hurdle = 0.03

    # Penalty for exchange disagreement (low congruence)
    if abs(congruence) < 1.0:
        hurdle += 0.04

    # Penalty for absorption (flow is fighting the book)
    if absorption < 0:
        hurdle += 0.02

    # 4. Execution Logic
    if best_ev > hurdle:
        # Multiplier 1: Scaling by directional consensus (alignment of OBI and Delta)
        # If all 3 exchanges agree, multiplier is 2.0.
        consensus_multiplier = 1.0 + (abs(congruence) / 3.0)

        # Multiplier 2: Squeeze Boost
        # If funding and delta are in opposition (negative squeeze_signal), 
        # we are in a high-conviction liquidation setup.
        squeeze_boost = 1.0
        if squeeze_signal < -0.00001:
            squeeze_boost = 1.25

        # Time-decay: Reduce bet size as we approach the 5-min resolution to avoid noise.
        time_factor = max(0.7, min(1.0, seconds_remaining / 300.0))

        # Final sizing: Scale by edge relative to hurdle, consensus, squeeze, and time.
        bet_fraction = (best_ev / hurdle) * consensus_multiplier * squeeze_boost * time_factor
        bet_fraction = min(1.0, max(0.0, bet_fraction))

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0