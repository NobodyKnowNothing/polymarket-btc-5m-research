# Var 12 | BT: 0.333 | Live PnL: $-6.31 | Trades: 60
# Desc: Implements "Signal-to-Noise Ratio (SNR) Adaptive Hurdle with Efficiency Gating," which uses the alignment between OBI and Delta to lower the entry hurdle, while using price-flow efficiency to raise th
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -0.7579123377799988, -1.0, -1.0, -1.0, 0.12663000000000002, 0.10316799999999993, 0.018941000000000097, 0.020114600658416748, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, 0.028705999999999898, -1.0, -1.0, -1.0, 1.5191249999999998, 0.12663000000000002, 2.651597, -1.0, 0.11476700000000006, -1.0, 1.3396370000000002, 4.893558, -0.8744395971298218, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 0.12663000000000002, 1.4562249999999999, 1.9791409999999998, 1.9791409999999998, -1.0, 1.1846420000000002, 2.073133, -1.0, 0.7858868837356567, -1.0, -1.0, 0.13876599999999994, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -6.313988208770752, "live_wins": 23, "live_total": 60, "age": 89, "rounds_underwater": 11, "abstain_rounds": 0, "backtest_score": 0.33340440893845846, "backtest_acc": 0.5600457491422036, "backtest_n": 4263}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure alignment (OBI/Delta) and price-flow efficiency.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # Ensure we are working with sorted time series within each market for diffs
    # (Assuming df is already sorted by timestamp, but we group by market_slug)
    
    # 1. Microstructure Alignment: Do OBI and Delta point in the same direction?
    # A value of 1 means perfect alignment, -1 means perfect opposition.
    X['obi_delta_alignment'] = np.sign(X['binance_delta']) * np.sign(X['binance_obi'])

    # 2. Price-Flow Efficiency: How much does price move per unit of aggressive delta?
    # We use the diff of price and delta within the market window.
    # High efficiency = trend is easy to follow. Low efficiency = absorption/friction.
    X['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    X['delta_change'] = df.groupby('market_slug')['binance_delta'].diff().fillna(0)
    
    # Efficiency = |delta_change| / |price_change| is too volatile. 
    # Let's use: |price_change| / (|delta_change| + epsilon)
    X['flow_efficiency'] = X['price_change'].abs() / (X['delta_change'].abs() + 1e-7)

    # 3. Volatility-Normalized Flow: Is the current delta significant relative to volatility?
    X['vol_norm_delta'] = X['binance_delta'] / (X['mean_volatility'] + 1e-7)

    # 4. Cross-Exchange Spread Stability: Is the price dispersion increasing?
    X['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using an SNR-Adaptive Hurdle.
    The hurdle is lowered when OBI and Delta align (high signal) 
    and raised when efficiency is low (high absorption/friction).
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered quality metrics
    # alignment: 1.0 (agree), -1.0 (oppose), 0.0 (neutral)
    alignment = features_row.get('obi_delta_alignment', 0.0)
    # efficiency: higher is smoother price movement
    efficiency = features_row.get('flow_efficiency', 1.0)

    # 3. Calculate Dynamic Hurdle
    # Base hurdle is 2.5%
    base_hurdle = 0.025
    
    # Alignment Bonus: If OBI and Delta agree, reduce hurdle by up to 1.5%
    alignment_bonus = 0.015 * max(0, alignment)
    
    # Efficiency Penalty: If efficiency is low (absorption), increase hurdle.
    # We normalize efficiency: an efficiency of 0.1 is "low", 1.0 is "normal".
    # We use a sigmoid-like penalty to avoid extreme jumps.
    efficiency_score = min(1.0, efficiency * 5.0) 
    efficiency_penalty = 0.03 * (1.0 - efficiency_score)

    dynamic_hurdle = base_hurdle - alignment_bonus + efficiency_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: As we approach the 5-min mark, reduce bet size to avoid noise spikes.
        # We scale from 100% at 300s down to 30% at 0s.
        time_factor = 0.3 + (0.7 * (seconds_remaining / 300.0))
        time_factor = max(0.3, min(1.0, time_factor))

        # Conviction Scaling: Scale bet size based on how much the EV exceeds the hurdle.
        # We use a soft-cap to prevent massive bets on single outliers.
        conviction = (best_ev / dynamic_hurdle)
        bet_fraction = min(1.0, conviction * time_factor)

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.15:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0