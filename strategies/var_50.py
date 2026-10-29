# Var 50 | BT: 0.273 | Live PnL: $-0.22 | Trades: 61
# Desc: Implements "Unified Impulse Capture (UIC)," which targets high-conviction breakouts by requiring strict confluence between aggressive flow (Delta), order book pressure (OBI), price efficiency, and cro
# Stats: {"live_returns": [-0.18459546566009521, -0.3355845510959625, -0.25331926345825195, -0.3242081105709076, -0.28838270902633667, -0.18581734597682953, -0.10408905893564224, -0.9272088408470154, -0.16422736644744873, 0.07205261290073395, -0.32381555438041687, -0.05950664356350899, -0.2437204122543335, -0.25393831729888916, -0.30198901891708374, -0.4949663579463959, 0.9031329154968262, -0.08216916024684906, -0.1164516806602478, -0.07402976602315903, -0.25610536336898804, -0.3774886131286621, 1.890848, -0.40665099024772644, 1.3579983711242676, -0.08492717891931534, -0.08986490219831467, -0.26117679476737976, -0.05578203126788139, 0.0171494260430336, -0.8000453114509583, 4.366225242614746, -0.38022512197494507, 3.1194439999999997, -0.1035904735326767, -0.2780117392539978, -0.2543969750404358, -0.3795503079891205, -0.09356918931007385, -0.5451935529708862, 0.6034295558929443, 0.11629023402929306, -0.2553558051586151, -0.18208113312721252, 0.36760884523391724, -0.07755014300346375, -0.28079235553741455, -0.10180597007274628, -0.13991700112819672, -0.10735655575990677, -0.24167658388614655, -0.30192840099334717, 0.4332798421382904, -0.0710664838552475, -0.8476550579071045, -0.3499080240726471, -0.20287209749221802, -0.17426182329654694, 0.016234681010246277, -1.0, -0.06481222063302994], "live_pnl": -0.21994349360466003, "live_wins": 12, "live_total": 61, "age": 87, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.2727363809289174, "backtest_acc": 0.5583482944344704, "backtest_n": 2342}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between flow, book pressure, and exchange agreement.
    """
    df = df.copy()

    # 1. Temporal Velocity: How fast are the signals changing?
    # Group by market_slug to prevent leakage.
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 2. Delta-OBI Sync: Do aggressive trades align with book pressure?
    # Positive = Flow and Book are in sync; Negative = Absorption.
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 3. Price Efficiency: How much price movement is generated per unit of delta?
    # High efficiency = Clean move; Low efficiency = Absorption/Noise.
    df['price_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 4. Exchange Agreement: How unified is the price across exchanges?
    # Higher is better (lower spread).
    df['exchange_agreement'] = 1.0 / (df['price_spread'] + 0.001)

    # 5. Momentum-Flow Sync: Is momentum backed by actual aggressive delta?
    df['momentum_delta_sync'] = df['mean_momentum'] * df['mean_delta']

    # Drop metadata but keep pm_costs and other features for the model
    # We drop seconds_remaining here because it is metadata, but it's passed to strategy_filter separately.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Unified Impulse Capture (UIC)".
    The strategy seeks 'clean' breakouts where delta, OBI, and price efficiency 
    all align, and where exchanges are in tight consensus.
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
        target_delta = features_row.get('mean_delta', 0.0)
        target_obi = features_row.get('mean_obi', 0.0)
    else:
        side = 'Down'
        best_ev = ev_down
        target_delta = -features_row.get('mean_delta', 0.0) # Invert for 'Down' side
        target_obi = -features_row.get('mean_obi', 0.0)    # Invert for 'Down' side

    # 2. Base Hurdle
    hurdle = 0.025
    if best_ev < hurdle:
        return None, 0.0

    # 3. Extract Microstructure Metrics
    efficiency = features_row.get('price_efficiency', 0.0)
    agreement = features_row.get('exchange_agreement', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    obi = features_row.get('mean_obi', 0.0)

    # 4. Multi-Factor Confluence Multiplier
    multiplier = 1.0

    # A. Directional Confluence: Do Delta and OBI actually support the 'side'?
    # If we want to go 'Up', we need positive delta and positive OBI.
    if (side == 'Up' and delta > 0 and obi > 0) or (side == 'Down' and delta < 0 and obi < 0):
        multiplier *= 1.6  # High conviction: Flow and Book are aligned
    else:
        multiplier *= 0.3  # Low conviction: Absorption or conflicting signals

    # B. Efficiency Check: Is the delta actually moving the price?
    if abs(efficiency) < 0.5:
        multiplier *= 0.4  # Avoid: High delta but no price movement (Absorption)
    elif abs(efficiency) > 2.0:
        multiplier *= 1.2  # Boost: Very efficient price movement

    # C. Consensus Check: Are exchanges agreeing on the price?
    # If agreement is low (high spread), the market is fragmented/noisy.
    if agreement < 10.0:
        multiplier *= 0.5  # Avoid: High exchange dispersion

    # 5. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid late-window volatility.
    time_factor = (seconds_remaining / 300.0)

    # Scale bet_fraction by (Edge / Hurdle) * Confluence * Time
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Final safety check: If the multiplier or edge is too low, abstain.
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction