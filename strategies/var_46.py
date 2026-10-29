# Var 46 | BT: -0.572 | Live PnL: $-1.75 | Trades: 61
# Desc: Implements "Microstructure-Regime Adaptive Sizing (MRAS)," which uses the ML model as the primary directional signal but dynamically scales bet sizes to reward "squeeze" setups (funding/delta divergen
# Stats: {"live_returns": [0.17833589017391205, -1.0, -0.9370791912078857, -0.09052839130163193, -1.0, -0.22150540351867676, 0.24343979358673096, -0.3502284288406372, 0.873223066329956, 0.03722783923149109, 0.5740789175033569, -0.5237036943435669, 2.356813430786133, -0.3918546140193939, -0.5649124979972839, 1.9084949493408203, -0.1681428700685501, -0.25995129346847534, -1.0, 0.5231292247772217, 0.2919268310070038, 0.26705169677734375, -0.24720630049705505, -0.42345109581947327, 0.01001971960067749, 0.8106191158294678, 0.5296780467033386, -0.5575854778289795, -0.13889269530773163, -1.0, -0.4424661099910736, -1.0, -0.561271071434021, 4.550646, 3.1194439999999997, -0.37844231724739075, -0.14592014253139496, -1.0, -1.0, -0.15606100857257843, 0.015930265188217163, -0.36059868335723877, 2.265998363494873, 0.014451950788497925, 0.4030195474624634, -0.3600241243839264, -1.0, 0.5364605188369751, -0.14674484729766846, -1.0, -0.35765695571899414, -0.6371335983276367, -0.8545215725898743, -1.0, 1.5175033807754517, 0.003546580672264099, -0.3683939576148987, -1.0, -1.0, -1.0, -0.13542644679546356], "live_pnl": -1.748663067817688, "live_wins": 22, "live_total": 61, "age": 91, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": -0.5718707013918835, "backtest_acc": 0.5366752672318467, "backtest_n": 2295}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the interaction between aggressive flow, passive book pressure, and 
    the cost of carry (funding).
    """
    df = df.copy()

    # 1. Delta-OBI Sync: Do aggressive trades align with the passive order book?
    # Positive = Buying into a bid-heavy book (Impulse); Negative = Buying into a thin/ask-heavy book (Absorption).
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 2. Funding-Delta Divergence: Measures potential squeeze setups.
    # Negative = Long squeeze (Positive funding, Negative delta) or Short squeeze (Negative funding, Positive delta).
    df['funding_delta_sync'] = df['mean_funding'] * df['mean_delta']

    # 3. Volatility-Adjusted Flow: How much delta is moving the price relative to current volatility.
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Binance Dominance: Re-include the strength of the primary liquidity driver.
    df['binance_delta_rel'] = df['binance_delta'] / (df['mean_delta'].abs() + 1e-6)

    # Drop metadata
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on signal over noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 45.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using "Microstructure-Regime Adaptive Sizing (MRAS)".
    The model provides the directional edge (EV), while the microstructure 
    determines the confidence multiplier (Squeeze vs. Absorption).
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

    # 2. Extract Microstructure Signals
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    mean_funding = features_row.get('mean_funding', 0.0)
    price_spread = features_row.get('price_spread', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    # 3. Base Hurdle
    base_hurdle = 0.025
    if best_ev < base_hurdle:
        return None, 0.0

    # 4. Regime-Based Multiplier
    multiplier = 1.0

    # A) SQUEEZE ACCELERATOR: Reward trades that align with funding-driven squeezes.
    # Long Squeeze: Funding is positive (longs pay), Delta is negative (selling). Price likely DOWN.
    is_long_squeeze = (mean_funding > 0.0001 and mean_delta < -0.05)
    # Short Squeeze: Funding is negative (shorts pay), Delta is positive (buying). Price likely UP.
    is_short_squeeze = (mean_funding < -0.0001 and mean_delta > 0.05)

    if side == 'Down' and is_long_squeeze:
        multiplier *= 1.6
    elif side == 'Up' and is_short_squeeze:
        multiplier *= 1.6

    # B) ABSORPTION GUARD: Penalize trades where aggressive flow hits a wall.
    # Buying into a book with negative OBI (more asks/selling pressure).
    is_buying_absorption = (mean_delta > 0.05 and mean_obi < -0.15)
    # Selling into a book with positive OBI (more bids/buying pressure).
    is_selling_absorption = (mean_delta < -0.05 and mean_obi > 0.15)

    if side == 'Up' and is_buying_absorption:
        multiplier *= 0.4
    elif side == 'Down' and is_selling_absorption:
        multiplier *= 0.4

    # C) FRICTION GUARD: Penalize high cross-exchange dispersion.
    if price_spread > (mean_vol * 3.0 + 1e-6):
        multiplier *= 0.5

    # 5. Execution & Sizing
    # Quadratic time decay to reduce exposure as the window closes.
    time_factor = (seconds_remaining / 300.0) ** 2
    
    # Scale bet by (Edge/Hurdle) and the regime multiplier.
    bet_fraction = (best_ev / base_hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trade significance
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction