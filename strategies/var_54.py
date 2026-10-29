# Var 54 | BT: -1.273 | Live PnL: $-23.99 | Trades: 60
# Desc: Implements "Noise-Adaptive Edge Gating (NAEG)," which scales the required EV hurdle and bet size based on a combined uncertainty metric of exchange price dispersion and volatility to avoid trading in
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, 0.0722506046295166, -1.0, -1.0, -1.0, -1.0, 2.39541, 0.8745183348655701, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, 0.09182599999999996, -1.0, 3.505625, -1.0, -1.0, -1.0, 0.06987699999999997, -1.0, 1.7294070000000001, -0.819351077079773, -1.0, -1.0, -1.0, 0.03115522861480713, -1.0, -1.0, 0.04885600000000001, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0], "live_pnl": -23.99322509765625, "live_wins": 14, "live_total": 60, "age": 124, "rounds_underwater": 60, "abstain_rounds": 1, "backtest_score": -1.2734191611380696, "backtest_acc": 0.5896076352067868, "backtest_n": 1328}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price efficiency and cross-exchange dispersion.
    """
    df = df.copy()

    # 1. Price Efficiency: How much delta is required to move the price.
    # Low efficiency means high absorption; high efficiency means high momentum.
    df['price_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-4)

    # 2. Exchange Dispersion: The relative spread between exchanges.
    # High dispersion suggests market uncertainty or localized liquidity gaps.
    df['exchange_dispersion'] = df['price_spread'] / (df['binance_price'] + 1e-4)

    # 3. Delta-OBI Alignment: Product of flow and book state.
    df['delta_obi_alignment'] = df['mean_delta'] * df['mean_obi']

    # Drop metadata but keep all pre-parsed features and our new engineered ones.
    # We keep 'seconds_remaining', 'pm_cost_up', and 'pm_cost_down' as they are useful features.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Noise-Adaptive Edge Gating (NAEG)".
    This strategy calculates an uncertainty score from price dispersion and volatility,
    then uses that score to dynamically increase the EV hurdle and penalize bet size.
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

    # 2. Calculate Uncertainty Score
    # Combines exchange disagreement (dispersion) and local volatility.
    dispersion = features_row.get('exchange_dispersion', 0.0)
    volatility = features_row.get('mean_volatility', 0.0)
    
    # Uncertainty is a weighted sum of dispersion and volatility
    uncertainty = (dispersion * 15.0) + (volatility * 0.5)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle + uncertainty penalty + time-decay penalty.
    # As uncertainty or time-remaining (approaching 0) increases, the hurdle rises.
    time_ratio = max(0.0, seconds_remaining / 300.0)
    time_penalty = (1.0 - time_ratio) * 0.02
    hurdle = 0.02 + uncertainty + time_penalty

    if best_ev < hurdle:
        return None, 0.0

    # 4. Sizing Logic: Scale by (Edge/Hurdle) and penalize by uncertainty.
    # We also check for microstructure congruence: does Delta and OBI support the side?
    mean_d = features_row.get('mean_delta', 0.0)
    mean_o = features_row.get('mean_obi', 0.0)
    
    congruence_multiplier = 1.0
    if side == 'Up':
        if mean_d < 0 or mean_o < 0:
            congruence_multiplier = 0.7  # Divergence penalty
    else:  # side == 'Down'
        if mean_d > 0 or mean_o > 0:
            congruence_multiplier = 0.7  # Divergence penalty

    # Bet fraction scales with edge but is heavily dampened by uncertainty.
    bet_fraction = (best_ev / hurdle) * (1.0 / (1.0 + uncertainty)) * congruence_multiplier
    
    # Final safety constraints
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction