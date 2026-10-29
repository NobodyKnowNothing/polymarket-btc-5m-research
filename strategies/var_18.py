# Var 18 | BT: 0.288 | Live PnL: $-26.38 | Trades: 61
# Desc: Implements "Cross-Exchange Delta-Consensus & Absorption-Gated Sizing," which uses the magnitude of directional agreement across exchanges to scale conviction and applies a dynamic hurdle to penalize a
# Stats: {"live_returns": [0.20391000000000004, 0.1769179999999999, 1.233902, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0933409999999999, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, 3.1194439999999997, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, -0.9333590865135193, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.13876599999999994, 1.1846420000000002, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -0.7313541769981384, -1.0, -1.0, 2.39541, 1.1450012922286987, -0.6693788766860962, -1.0, 1.4562249999999999, -1.0, -1.0, -0.6693788766860962, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.801360547542572], "live_pnl": -26.384145736694336, "live_wins": 14, "live_total": 61, "age": 85, "rounds_underwater": 57, "abstain_rounds": 0, "backtest_score": 0.28807872851649685, "backtest_acc": 0.5264693622342643, "backtest_n": 1836}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus and microstructure tension.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Delta Consensus: How many exchanges agree on the direction of aggressive flow
    # Range: [-3, 3]. 3 = all bullish, -3 = all bearish, 0 = mixed/neutral.
    X['delta_consensus'] = (np.sign(X['binance_delta']) + 
                            np.sign(X['coinbase_delta']) + 
                            np.sign(X['kraken_delta']))

    # 2. Microstructure Tension (Absorption): High OBI but negative Delta (or vice versa)
    # A negative value indicates "absorption" (aggressive sellers hitting high bids).
    X['obi_delta_tension'] = X['mean_obi'] * X['mean_delta']

    # 3. Price-Flow Cohesion: Alignment between price movement and delta direction
    # 1.0 = price and delta move together; -1.0 = divergence (potential reversal/absorption).
    X['price_flow_cohesion'] = np.sign(X['price_vs_strike_pct']) * np.sign(X['mean_delta'])

    # 4. Delta Acceleration: The second-order derivative of flow
    # We must group by market_slug to ensure we don't calculate diffs across different markets.
    X['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().diff().fillna(0)

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
    Decide whether to trade using a Consensus-Weighted Hurdle.
    The hurdle increases during exchange disagreement or market absorption.
    Bet size is amplified by the strength of cross-exchange consensus.
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
    consensus = features_row.get('delta_consensus', 0.0)
    tension = features_row.get('obi_delta_tension', 0.0)

    # 3. Calculate Dynamic Hurdle
    # Base hurdle is 3%.
    hurdle = 0.03

    # Penalty for exchange disagreement (if consensus is near 0)
    if abs(consensus) < 1.0:
        hurdle += 0.04
    
    # Penalty for absorption (tension < 0 means OBI and Delta are fighting)
    if tension < 0:
        hurdle += 0.02

    # Penalty for last-minute volatility/noise
    if seconds_remaining < 60:
        hurdle += 0.02

    # 4. Execution Logic
    if best_ev > hurdle:
        # Consensus Multiplier: Scales bet size based on how much exchanges agree.
        # If consensus is 3 (perfect), multiplier is 2.0. 
        # If consensus is 0 (mixed), multiplier is 1.0.
        consensus_multiplier = 1.0 + (abs(consensus) / 3.0)
        
        # Time-decay: Reduce bet size as we approach the 5-min resolution.
        time_factor = max(0.5, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale linearly with the edge relative to the hurdle, 
        # then amplify by consensus and time.
        bet_fraction = (best_ev / hurdle) * consensus_multiplier * time_factor
        bet_fraction = min(1.0, max(0.0, bet_fraction))

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0