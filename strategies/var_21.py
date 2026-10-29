# Var 21 | BT: 0.084 | Live PnL: $-20.00 | Trades: 61
# Desc: Implements "Directional Microstructure Alignment & Volatility-Scaled Conviction (DMAVC)," which uses the sign-congruence of Order Book Imbalance and Aggressive Delta to amplify or dampen the model's p
# Stats: {"live_returns": [2.173394, -1.0, 1.5854279999999998, 1.1846420000000002, -1.0, -0.9715500473976135, -0.6261765360832214, 0.7534775137901306, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7119762897491455, -1.0, -1.0, 0.0933409999999999, -1.0, -0.9938075542449951, -1.0, 1.5191249999999998, -0.5269374847412109, -0.4964788556098938, -1.0, -1.0, -1.0, -1.0, 0.07477504014968872, -1.0, 4.550646, 3.1194439999999997, -0.9694833755493164, -1.0, -1.0, -0.672863245010376, -0.5942378640174866, 2.39541, -0.7634112238883972, -1.0, 1.396601676940918, -0.9202384352684021, -1.0, -1.0, -1.0, -0.6187084317207336, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -0.5218283534049988, -1.0, -1.0, -1.0, -0.6322743892669678, 2.073133, 0.05111473798751831, -0.9277853965759277], "live_pnl": -19.998083114624023, "live_wins": 14, "live_total": 61, "age": 99, "rounds_underwater": 51, "abstain_rounds": 0, "backtest_score": 0.0836889218430722, "backtest_acc": 0.5388773388773389, "backtest_n": 1736}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronization between order book state and aggressive flow.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Delta-OBI Synchronization: Do the aggressive trades align with the book imbalance?
    # Result is 1 (both positive), -1 (both negative), or 0 (mixed/neutral)
    X['delta_obi_sync'] = np.sign(X['sum_delta']) * np.sign(X['mean_obi'])

    # 2. Price-Flow Synchronization: Is the price actually moving in the direction of the delta?
    # This helps identify absorption (low sync) vs momentum (high sync).
    X['price_flow_sync'] = np.sign(X['sum_delta']) * np.sign(X['price_vs_strike_pct'])

    # 3. Market Implied Probability Context
    # Including costs allows the model to learn the relationship between 
    # model confidence and market pricing (the "edge").
    
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
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Directional Microstructure Alignment.
    The strategy amplifies the edge when OBI and Delta are congruent with the 
    predicted direction and scales the hurdle based on market volatility.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        edge = ev_up
        target_sign = 1
    else:
        side = 'Down'
        edge = ev_down
        target_sign = -1

    # 2. Microstructure Alignment Factor
    # We look at the 'delta_obi_sync' engineered feature.
    # If sync is 1 and we are going 'Up', alignment is high.
    # If sync is -1 and we are going 'Up', alignment is low.
    micro_sync = features_row.get('delta_obi_sync', 0.0)
    
    # alignment range: [0.5, 1.5]
    # If micro_sync (sign of delta * sign of obi) matches target_sign, boost edge.
    alignment = 1.0 + (micro_sync * target_sign * 0.5)

    # 3. Volatility-Adjusted Hurdle
    # Higher volatility requires a higher margin of safety.
    base_hurdle = 0.025
    vol_penalty = features_row.get('mean_volatility', 0.0) * 0.05
    dynamic_hurdle = base_hurdle + vol_penalty

    # 4. Execution Logic
    adjusted_edge = edge * alignment

    if adjusted_edge > dynamic_hurdle:
        # Time-decay: Drastically reduce bet size as we approach the 5-min resolution
        # to avoid being caught in the high-variance "noise" of the final seconds.
        time_factor = max(0.1, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale linearly with the strength of the adjusted edge relative to hurdle.
        bet_fraction = (adjusted_edge / dynamic_hurdle) * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure trades are meaningful
        if bet_fraction < 0.05:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0