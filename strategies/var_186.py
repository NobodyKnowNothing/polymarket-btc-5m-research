# Var 186 | BT: -0.727 | Live PnL: $-23.66 | Trades: 60
# Desc: Implements "Microstructure-Price Divergence (MPD) Lag-Capture," which targets the "sweet spot" where model edge, order book pressure, and exchange consensus align, but the price has not yet moved sign
# Stats: {"live_returns": [3.1194439999999997, -1.0, 1.9791409999999998, 2.280573, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, 0.01866292953491211, 0.018342912197113037, -1.0, -1.0, -1.0, 0.11476700000000006, 0.07390952110290527, -1.0, 0.08073200000000003, -1.0, -1.0, 0.03353208303451538, -1.0, -1.0, -1.0, -1.0, 0.12663000000000002, -1.0, -1.0, 0.07711052894592285, -0.8808896541595459, 0.017920613288879395, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9005675911903381, 2.280573, -1.0, 0.0652891993522644, -1.0, -1.0, -1.0, -1.0, 0.6888001561164856, -1.0, -1.0, -1.0, 1.1846420000000002, 1.890848, -1.0, 1.890848, -1.0, 0.06987699999999997, -1.0, -1.0, -1.0], "live_pnl": -23.655047087524416, "live_wins": 20, "live_total": 60, "age": 329, "rounds_underwater": 47, "abstain_rounds": 0, "backtest_score": -0.7265130171216757, "backtest_acc": 0.5661461138036259, "backtest_n": 983}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure pressure and cross-exchange consensus.
    """
    X = df.copy()

    # 1. Microstructure Pressure: The product of aggressive flow and book imbalance
    # High positive = aggressive buying into a heavy bid book.
    X['micro_pressure'] = X['mean_delta'] * X['mean_obi']

    # 2. OBI Agreement: How many exchanges show the same directional intent?
    # Range [-3, 3]
    X['obi_agreement'] = (
        np.sign(X['binance_obi']) + 
        np.sign(X['coinbase_obi']) + 
        np.sign(X['kraken_obi'])
    )

    # 3. Delta Efficiency: How much price movement is generated per unit of delta?
    # Helps distinguish between 'sweeps' (high efficiency) and 'absorption' (low efficiency)
    X['delta_vol_ratio'] = X['mean_delta'].abs() / (X['mean_volatility'] + 1e-6)

    # Drop metadata and target
    # We must drop 'seconds_remaining' because it is passed separately to the filter
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 50.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Price Divergence (MPD)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Identify "Lag-Capture" opportunities:
       - Model predicts direction.
       - Microstructure pressure (Delta * OBI) aligns with prediction.
       - Exchange consensus (OBI agreement) is non-zero.
       - Price distance from strike is small (avoiding chasing an exhausted move).
    3. Dynamic Sizing:
       - Scale by EV magnitude.
       - Scale down as price approaches the 'exhaustion zone' (price_vs_strike_pct).
       - Scale down as the market window closes to avoid last-second volatility.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    micro_pressure = features_row.get('micro_pressure', 0.0)
    obi_agreement = features_row.get('obi_agreement', 0.0)
    price_dist = abs(features_row.get('price_vs_strike_pct', 0.0))
    
    side = None
    best_ev = 0.0
    
    # Thresholds
    EV_THRESHOLD = 0.025
    EXHAUSTION_LIMIT = 0.012  # 1.2% distance from strike is the limit
    
    # 1. Directional Logic with Microstructure Gating
    # We only enter if the model edge is high AND the microstructure supports it
    # AND we aren't already too far from the strike.
    if ev_up > EV_THRESHOLD and micro_pressure > 0 and obi_agreement >= 1 and price_dist < EXHAUSTION_LIMIT:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > EV_THRESHOLD and micro_pressure < 0 and obi_agreement <= -1 and price_dist < EXHAUSTION_LIMIT:
        side = 'Down'
        best_ev = ev_down

    # 2. Execution and Dynamic Sizing
    if side:
        # Base bet is scaled by the edge (targeting 4% EV for full size)
        base_bet = (best_ev / 0.04)
        
        # Exhaustion Penalty: Linear decay as price moves towards the exhaustion limit
        # If price_dist is 0, multiplier is 1.0. If price_dist is 0.012, multiplier is 0.0.
        exhaustion_multiplier = max(0.0, 1.0 - (price_dist / EXHAUSTION_LIMIT))
        
        # Time Decay: Reduce size in the final 60 seconds
        time_multiplier = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        bet_fraction = base_bet * exhaustion_multiplier * time_multiplier
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0