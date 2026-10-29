# Var 137 | BT: 1.630 | Live PnL: $-11.11 | Trades: 60
# Desc: Implements "Microstructure Convergence-Coupling (MCC)," which requires that directional order flow (Delta $\times$ OBI) and price momentum are not only aligned but also supported by high cross-exchang
# Stats: {"live_returns": [1.5191249999999998, -1.0, 2.280573, 1.890848, -1.0, -1.0, 1.3396370000000002, 1.7294070000000001, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 0.16389999999999993, -1.0, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, -1.0, 1.4562249999999999, 1.9791409999999998, 1.9791409999999998, -1.0, -1.0, 1.28551, 1.890848, -1.0, -1.0, -1.0, -1.0, 2.518758, -1.0, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0], "live_pnl": -11.112733, "live_wins": 17, "live_total": 60, "age": 441, "rounds_underwater": 40, "abstain_rounds": 0, "backtest_score": 1.6297955843868075, "backtest_acc": 0.5791691751956652, "backtest_n": 899}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coupling between order flow, price movement, and exchange unanimity.
    """
    X = df.copy()

    # 1. Flow Force: Combined strength of aggressive delta and order book pressure.
    # Positive = aggressive buying into a bid-heavy book.
    X['flow_force'] = X['mean_delta'] * X['mean_obi']

    # 2. Price Momentum: Rate of price change relative to remaining time.
    # Uses price_vs_strike_pct to capture the velocity of the move.
    X['price_momentum'] = X['price_vs_strike_pct'] / (X['seconds_remaining'] + 10.0)

    # 3. Coupling Index: Measures if the order flow is actually translating into price action.
    # High positive = flow and price are moving together bullishly.
    # High negative = flow and price are moving together bearishly.
    X['coupling_index'] = X['flow_force'] * X['price_momentum']

    # 4. Unanimity Score: Measures how much the exchanges agree.
    # Low price_spread and low obi_spread = high unanimity (near 1.0).
    # High dispersion = low unanimity (near 0.0).
    X['unanimity_score'] = 1.0 / (1.0 + X['price_spread'] + X['obi_spread'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Convergence-Coupling (MCC)".

    Logic:
    - Calculate raw EV for both sides.
    - Alignment Check: Does the direction of the model's edge (Up/Down) 
      match the direction of the Microstructure Coupling (flow + price)?
    - Unanimity Gating: Use the 'unanimity_score' to dynamically adjust the 
      required EV threshold. If exchanges disagree (low unanimity), we 
      require a much higher edge to enter.
    - Bet Sizing: Scale the bet fraction by the strength of the coupling 
      and the magnitude of the edge.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    coupling = features_row.get('coupling_index', 0.0)
    unanimity = features_row.get('unanimity_score', 0.0)

    # 1. Alignment Check
    # If we are betting 'Up', we want coupling (flow * momentum) to be positive.
    is_aligned = (side == 'Up' and coupling > 0) or (side == 'Down' and coupling < 0)
    
    if not is_aligned:
        return None, 0.0

    # 2. Dynamic Thresholding via Unanimity
    # If unanimity is 1.0, threshold is ~0.02. 
    # If unanimity is 0.1, threshold is ~0.06.
    # This forces us to only trade low-spread, high-agreement markets.
    dynamic_threshold = 0.025 / (unanimity + 0.2)

    if best_ev > dynamic_threshold:
        # 3. Dynamic Bet Sizing
        # Scale by the edge and the strength of the coupling (capped to prevent over-leveraging).
        # We target a 4% edge for a full bet.
        coupling_multiplier = 1.0 + min(1.0, abs(coupling))
        bet_fraction = (best_ev / 0.04) * coupling_multiplier
        
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0