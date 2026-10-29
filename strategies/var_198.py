# Var 198 | BT: -0.056 | Live PnL: $0.16 | Trades: 6
# Desc: Implements "Temporal Microstructure Consensus (TMC)," which scales entry hurdles and bet sizes based on the decay of microstructure signal relevance over the 5-minute window and the consensus between
# Stats: {"live_returns": [-0.35192332310438157, 0.5333035758749716, -0.42167007692480085, 0.7801155165888278, 0.030003406517693232, -0.40575384346636134], "live_pnl": 0.16407525548594892, "live_wins": 3, "live_total": 6, "age": 170, "rounds_underwater": 0, "abstain_rounds": 59, "backtest_score": -0.05580519796326019, "backtest_acc": 0.5565602836879433, "backtest_n": 474}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-venue consensus and volatility-normalized flow.
    """
    df = df.copy()

    # 1. OBI Consensus: How much do Binance and Coinbase agree on order book imbalance?
    # Range [0, 1], where 1 is perfect agreement.
    obi_diff = (df['binance_obi'] - df['coinbase_obi']).abs()
    df['obi_consensus'] = np.clip(1.0 - obi_diff, 0, 1)

    # 2. Delta Consensus: Do the aggressive flows on both exchanges point in the same direction?
    # Range [0, 1], where 1 is perfect agreement.
    delta_sign_diff = (np.sign(df['binance_delta']) - np.sign(df['coinbase_delta'])).abs()
    df['delta_consensus'] = np.clip(1.0 - delta_sign_diff, 0, 1)

    # 3. Volatility-Normalized Delta: Signal strength relative to price noise
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Microstructure Sentiment: A composite of OBI and Delta direction
    # Range [-1, 1]
    df['micro_sentiment'] = (df['mean_obi'] + np.sign(df['mean_delta'])) / 2.0

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
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Temporal Microstructure Consensus (TMC)" logic.

    Logic:
    1. Time Decay: Microstructure signals (OBI/Delta) are most predictive early in the 
       5-minute window. As seconds_remaining approaches 0, the 'alpha' of these 
       signals decays.
    2. Consensus Filter: Only trade when Binance and Coinbase show high agreement 
       on both OBI and Delta direction.
    3. Dynamic Hurdle: The required Expected Value (EV) increases as the window 
       closes to avoid high-variance "last-second" gambling.
    4. Sizing: Bet size is scaled by both the level of consensus and the time remaining.
    """
    # Exit if too little time remains to allow for price movement
    if seconds_remaining < 30:
        return None, 0.0

    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # 2. Calculate Time Factor (1.0 at start of window, ~0.0 at end)
    # Assuming 300s window
    time_factor = min(1.0, seconds_remaining / 300.0)

    # 3. Extract engineered consensus features
    obi_con = features_row.get('obi_consensus', 0.5)
    delta_con = features_row.get('delta_consensus', 0.5)
    agreement_score = (obi_con + delta_con) / 2.0

    # 4. Determine Dynamic Hurdle
    # Hurdle starts at 0.02 and increases to 0.04 as time runs out
    # Also scales slightly with volatility to ensure edge covers noise
    base_vol_adj = 1.0 + features_row.get('mean_volatility', 0.5)
    dynamic_hurdle = 0.02 * (2.0 - time_factor) * base_vol_adj

    # 5. Decision Logic
    side = None
    bet_fraction = 0.0

    # We only trade if the consensus is high (agreement > 0.6)
    # and the EV exceeds our time-decaying hurdle.
    if agreement_score > 0.6:
        if ev_up > ev_down and ev_up > dynamic_hurdle:
            side = 'Up'
            # Scale bet by consensus and how much time is left to realize the edge
            bet_fraction = agreement_score * time_factor
        elif ev_down > ev_up and ev_down > dynamic_hurdle:
            side = 'Down'
            bet_fraction = agreement_score * time_factor

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0