# Var 52 | BT: 0.053 | Live PnL: $-13.48 | Trades: 61
# Desc: Implements "Dynamic Hurdle & Microstructure Congruence (DHMC)," which scales the required edge threshold based on time-to-resolution and uses OBI/Delta alignment to weight conviction, favoring trades
# Stats: {"live_returns": [-1.0, -1.0, 3.1194439999999997, -0.3980889320373535, -1.0, -0.4179689288139343, -1.0, 1.890848, -1.0, 0.2734881043434143, 4.245841, -0.5580859780311584, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -0.8684131503105164, -1.0, -1.0, -0.26075443625450134, -1.0, 0.1012687087059021, -1.0, -0.18764802813529968, -1.0, 0.13876599999999994, -1.0, 4.550646, 3.1194439999999997, -0.9418830871582031, -1.0, -1.0, -0.792889416217804, -1.0, 2.39541, 0.7169593572616577, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -0.3812316358089447, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -0.35643598437309265, -1.0, -0.6745971441268921, 0.0870370864868164, -1.0, -0.3095594644546509, -1.0, 0.0363294780254364, -0.4261105954647064], "live_pnl": -13.48367691040039, "live_wins": 15, "live_total": 61, "age": 86, "rounds_underwater": 43, "abstain_rounds": 0, "backtest_score": 0.05269619683387372, "backtest_acc": 0.5469140206921156, "backtest_n": 2390}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure alignment (Delta-OBI product) and price velocity.
    """
    df = df.copy()

    # 1. Microstructure Alignment: The product of delta and OBI.
    # Positive values suggest aggressive buying into a bid-heavy book.
    df['delta_obi_product'] = df['mean_delta'] * df['mean_obi']

    # 2. Volatility-Normalized Flow: Helps the model understand the 'strength' of delta.
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price Velocity: The rate of change in the percentage distance from strike.
    # We group by market_slug to ensure diffs are only calculated within the same window.
    df['price_velocity'] = df.groupby('market_slug')['price_vs_strike_pct'].diff().fillna(0)

    # Drop metadata but keep all pre-parsed features and our new engineered ones.
    # We keep 'seconds_remaining', 'pm_cost_up', and 'pm_cost_down' as they are useful features.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Dynamic Hurdle & Microstructure Congruence (DHMC)".
    This strategy uses a time-decaying hurdle (requiring more edge as the window closes)
    and scales bet size based on whether the order book (OBI) and flow (Delta) 
    congruently support the model's predicted direction.
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

    # 2. Dynamic Hurdle: Increase the required edge as the market approaches resolution.
    # At 300s, hurdle is ~0.02. At 0s, hurdle is ~0.04.
    time_ratio = max(0.0, seconds_remaining / 300.0)
    hurdle = 0.02 + (1.0 - time_ratio) * 0.02

    if best_ev < hurdle:
        return None, 0.0

    # 3. Microstructure Congruence Multiplier
    # We check if the flow (delta) and the book (obi) agree with the model's side.
    mean_d = features_row.get('mean_delta', 0.0)
    mean_o = features_row.get('mean_obi', 0.0)
    
    congruence_multiplier = 0.5  # Default: low conviction if no agreement
    
    if side == 'Up':
        if mean_d > 0 and mean_o > 0:
            congruence_multiplier = 1.3  # Strong agreement
        elif mean_d > 0 or mean_o > 0:
            congruence_multiplier = 1.0  # Partial agreement
    else:  # side == 'Down'
        if mean_d < 0 and mean_o < 0:
            congruence_multiplier = 1.3  # Strong agreement
        elif mean_d < 0 or mean_o < 0:
            congruence_multiplier = 1.0  # Partial agreement

    # 4. Sizing Logic
    # Scale by (Edge/Hurdle), the congruence, and the time remaining.
    # We use time_ratio to reduce exposure as the window closes to avoid last-second noise.
    bet_fraction = (best_ev / hurdle) * congruence_multiplier * time_ratio
    
    # Final safety constraints
    bet_fraction = min(1.0, max(0.0, bet_fraction))
    
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction