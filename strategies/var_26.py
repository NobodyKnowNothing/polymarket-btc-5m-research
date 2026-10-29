# Var 26 | BT: -0.489 | Live PnL: $-15.79 | Trades: 61
# Desc: Implements "Cross-Exchange Convergence & Fragmentation Gating (CECFG)," which uses the relationship between Binance price leadership and cross-exchange price dispersion to differentiate between high-c
# Stats: {"live_returns": [-0.8786816000938416, -1.0, -1.0, -1.0, 1.5191249999999998, 3.1194439999999997, 1.6554160000000002, -1.0, -0.8426633477210999, -1.0, -0.669679582118988, -0.668214738368988, -1.0, 1.890848, 1.0653259754180908, -1.0, -1.0, -0.6190477609634399, -1.0, -1.0, -1.0, -0.9809404611587524, -1.0, -0.7711381316184998, -1.0, -0.6827632784843445, -1.0, -1.0, -0.4790654182434082, 0.09570461511611938, 1.1846420000000002, -1.0, 4.550646, 3.1194439999999997, -0.9103146195411682, 0.0536838173866272, -1.0, -1.0, -1.0, -0.8489355444908142, -0.8010729551315308, -1.0, -1.0, -1.0, 2.39541, 0.03224211931228638, 1.2287852764129639, -1.0, -1.0, 1.0162501335144043, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0], "live_pnl": -15.790183067321777, "live_wins": 16, "live_total": 61, "age": 74, "rounds_underwater": 46, "abstain_rounds": 0, "backtest_score": -0.4890224422030143, "backtest_acc": 0.562807881773399, "backtest_n": 1988}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    exchange leadership and price dispersion dynamics.
    """
    # Drop metadata but keep columns needed for feature engineering
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Binance Leadership: How much Binance price deviates from the average of others
    # A positive value means Binance is leading the move higher.
    X['binance_lead_price'] = X['binance_price'] - (X['coinbase_price'] + X['kraken_price']) / 2.0

    # 2. Dispersion Velocity: Is the cross-exchange spread narrowing or widening?
    # We use groupby to ensure diffs are only calculated within the same market window.
    X['price_spread_diff'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 3. Order-Flow/Book Sync: Alignment between OBI and aggressive delta
    X['obi_delta_sync'] = X['mean_obi'] * X['sum_delta']

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Cross-Exchange Convergence Gating.
    
    High conviction is placed on 'Convergence' (when the leader's price move 
    is being matched by other exchanges, narrowing the spread).
    High risk is assigned to 'Fragmentation' (when the leader moves and 
    others diverge, increasing the spread).
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

    # 2. Extract engineered features
    lead = features_row.get('binance_lead_price', 0.0)
    spread_change = features_row.get('price_spread_diff', 0.0)
    sync = features_row.get('obi_delta_sync', 0.0)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 3%
    hurdle = 0.03

    # Convergence Logic: If spread is narrowing (spread_change < 0), it's a high-conviction move.
    # If spread is widening (spread_change > 0), it's a fragmentation/liquidity-gap risk.
    if abs(lead) > 0:
        if spread_change < 0:
            # Convergence: The market is coming together. Lower the hurdle.
            hurdle *= 0.7 
        else:
            # Fragmentation: The market is splitting. Raise the hurdle significantly.
            hurdle *= 1.6

    # Flow/Book Agreement Penalty: If OBI and Delta are fighting, increase hurdle.
    if abs(sync) < 0.05:
        hurdle += 0.02

    # 4. Execution Logic
    if best_ev > hurdle:
        # Time-decay: Reduce bet size as we approach the 5-min resolution.
        # We also stop trading entirely in the final 20 seconds to avoid noise.
        if seconds_remaining < 20:
            return None, 0.0
            
        time_factor = max(0.3, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale linearly with the strength of the edge relative to the hurdle.
        bet_fraction = (best_ev / hurdle) * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0