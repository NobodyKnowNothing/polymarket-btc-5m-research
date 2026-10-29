# Var 8 | BT: -0.210 | Live PnL: $-0.71 | Trades: 60
# Desc: Implements "Directional Efficiency Gating," which distinguishes between efficient trends (price moves in sync with flow) and absorption (price moves against flow), applying a multiplier to the expecte
# Stats: {"live_returns": [-0.25217965245246887, -0.4674440876642863, -0.4014058295885722, -0.38665640354156494, 0.5122432099397778, 1.4641574510586897, -0.24714350700378418, -0.46886364618937176, -0.2771523594856262, -0.4690080761909485, -0.46858915011088054, 0.4323228597640991, 0.04152020812034607, 0.03767430782318115, -0.4690581242243449, 0.010002046823501587, 1.2438260023100425, -0.4684545381863912, -0.46956255356470744, -0.26837924122810364, -0.46657932360967, -0.41026654839515686, 0.43938079476356506, 0.004393026232719421, 1.2385289940783557, 0.5370157250017382, 2.2946272851319645, -0.3317328691482544, -0.4025901198387146, -0.4565813406308492, -0.24536974728107452, -0.4622986614704132, -0.46537439107894896, -0.4694508210817973, -0.46951326449712116, 0.9734827354576286, 0.6825139804244635, 0.5773344189086135, 0.9260813864097897, -0.46935467004776, 0.44770362973213196, 0.9327925407930907, -0.3691501832008362, 0.5332896243269762, -0.35297961950302126, -0.46957515319188436, -0.4670796084403992, -0.46947721163431805, 0.8423767304082297, -0.469630126953125, -0.4697843805948893, -0.46880321820576987, -0.4675597874323527, -0.4696924662590027, 0.6552181566128479, -0.452253737449646, -0.4329928159713745, 0.024270594120025635, -0.4698598639170329, 0.03269690890889004], "live_pnl": -0.7083948254585266, "live_wins": 23, "live_total": 60, "age": 98, "rounds_underwater": 4, "abstain_rounds": 0, "backtest_score": -0.20954096608759754, "backtest_acc": 0.5719992391097585, "backtest_n": 3200}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features by calculating 
    the efficiency of price movement relative to aggressive flow.
    """
    X = df.copy()

    # 1. Price Change: Absolute movement from the start of the market window
    X['price_change'] = X['binance_price'] - X['strike_price']

    # 2. Delta Efficiency: How much price movement we get per unit of aggressive flow.
    # We use tanh to squash this into a [0, 1] range.
    # High efficiency = price moves easily with flow (Trend).
    # Low efficiency = price barely moves despite high flow (Absorption).
    X['delta_efficiency'] = np.tanh(X['price_change'].abs() / (X['mean_delta'].abs() + 1e-6))

    # 3. Flow Alignment: Do the sign of the delta and the sign of the price change match?
    # 1 = Trend (Flow and Price moving together)
    # -1 = Absorption/Reversal (Flow and Price moving in opposite directions)
    # 0 = Neutral (No significant movement or flow)
    X['flow_alignment'] = np.sign(X['mean_delta']) * np.sign(X['price_change'])

    # Drop metadata and target columns
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup: handle NaNs and infinities
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for high regularization to prevent overfitting noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 8.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using Directional Efficiency Gating.
    Amplify EV when price movement is efficient and aligned with flow; 
    drastically penalize EV when price movement suggests absorption.
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

    # 2. Extract engineered gating features
    alignment = features_row.get('flow_alignment', 0.0)
    efficiency = features_row.get('delta_efficiency', 0.0)

    # 3. Calculate Conviction Multiplier
    # If alignment is 1 (Trend): multiplier is [1.0, 2.0]
    # If alignment is -1 (Absorption): multiplier is [0.0, 1.0]
    # If alignment is 0 (Neutral): multiplier is 1.0
    if alignment > 0:
        multiplier = 1.0 + efficiency
    elif alignment < 0:
        multiplier = 1.0 - efficiency
    else:
        multiplier = 1.0

    effective_ev = best_ev * multiplier

    # 4. Execution Logic
    # Base hurdle for effective edge
    hurdle = 0.03

    if effective_ev > hurdle:
        # Time-decay: Reduce exposure as the window closes to mitigate last-second noise
        time_factor = max(0.3, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale based on how much the effective edge exceeds the hurdle
        # We use a conservative 0.4 scaling coefficient
        bet_fraction = min(1.0, (effective_ev / hurdle) * 0.4) * time_factor

        # Minimum bet threshold to avoid dust
        if bet_fraction < 0.02:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0