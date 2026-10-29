# Var 58 | BT: -0.353 | Live PnL: $-10.01 | Trades: 61
# Desc: Implements "Triadic Momentum-Cost Lag (TMCL)," which targets mispricing by identifying when the velocity of the actual BTC price (in percentage terms) and aggressive order flow outpace the adjustment
# Stats: {"live_returns": [0.0413166880607605, 0.05512362718582153, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.3964720000000002, 1.3396370000000002, -1.0, 5.282196, -1.0, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -0.3687157928943634, -1.0, 0.06987699999999997, 0.038675000000000015, 3.1194439999999997, -1.0, -1.0, -0.5365970134735107, -1.0, 0.5770750641822815, -1.0, -1.0, 0.8074487447738647, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 2.39541, 1.28551, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0444922149181366, -0.7877565622329712], "live_pnl": -10.009057998657227, "live_wins": 19, "live_total": 61, "age": 87, "rounds_underwater": 15, "abstain_rounds": 0, "backtest_score": -0.3532884438868992, "backtest_acc": 0.5314422419685577, "backtest_n": 2547}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the velocity divergence between exchange prices and Polymarket costs.
    """
    df = df.copy()

    # 1. Price Velocity (Percentage change to keep scale consistent)
    df['price_vel_pct'] = df.groupby('market_slug')['binance_price'].pct_change().fillna(0)
    
    # 2. Cost Velocity (Absolute change in contract price)
    df['cost_up_vel'] = df.groupby('market_slug')['pm_cost_up'].diff().fillna(0)
    df['cost_down_vel'] = df.groupby('market_slug')['pm_cost_down'].diff().fillna(0)
    
    # 3. Flow Velocity
    df['delta_vel'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)

    # 4. Lag Features: Does the price move faster than the Polymarket cost?
    # Positive lag means the 'real' price is moving faster than the prediction market.
    df['cost_lag_up'] = df['price_vel_pct'] - df['cost_up_vel']
    df['cost_lag_down'] = (-df['price_vel_pct']) - df['cost_down_vel']

    # 5. Microstructure Sync: Alignment of price velocity and aggressive flow
    df['price_flow_sync'] = df['price_vel_pct'] * df['sum_delta']

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 10.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Triadic Momentum-Cost Lag (TMCL)".
    The hurdle is lowered when the model's predicted edge is supported by 
    a 'lag' (price moving faster than cost) and 'sync' (flow and price alignment).
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
        lag_val = features_row.get('cost_lag_up', 0.0)
        sync_val = features_row.get('price_flow_sync', 0.0)
    else:
        side = 'Down'
        best_ev = ev_down
        lag_val = features_row.get('cost_lag_down', 0.0)
        sync_val = -features_row.get('price_flow_sync', 0.0) # Invert for Down side

    # 2. Minimum EV threshold to even consider a trade
    if best_ev < 0.015:
        return None, 0.0

    # 3. Calculate Confluence and Lag scores
    # Confluence: Are price and delta moving in the same direction?
    confluence = 1.0 if sync_val > 0 else 0.0
    
    # Lag Score: How much is the cost lagging behind the price move?
    # We normalize lag relative to a 10bps (0.001) move.
    lag_score = max(0.0, lag_val / 0.001)

    # 4. Dynamic Hurdle Calculation
    # Base hurdle is 3.5%. 
    # If confluence is high and lag is high, the hurdle drops significantly.
    # Max reduction cap: hurdle can drop to ~0.7% if lag and confluence are extreme.
    hurdle = 0.035 / (1.0 + confluence + min(lag_score, 5.0))

    # 5. Execution Decision
    if best_ev < hurdle:
        return None, 0.0

    # 6. Dynamic Sizing
    # Scale by edge strength, time remaining, and the strength of the lag.
    time_factor = max(0.1, seconds_remaining / 300.0)
    edge_strength = best_ev / hurdle
    
    # We slightly increase bet size if the lag is very high (high conviction mispricing)
    lag_multiplier = 1.0 + min(lag_score * 0.1, 0.5)
    
    bet_fraction = edge_strength * time_factor * lag_multiplier
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to avoid dust
    if bet_fraction < 0.2:
        return None, 0.0

    return side, bet_fraction