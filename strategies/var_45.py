# Var 45 | BT: -0.252 | Live PnL: $-11.20 | Trades: 61
# Desc: Synergistic Flow-Book Alignment (SFBA) — Targets sustainable moves by requiring alignment between aggressive trade flow (Delta) and order book pressure (OBI), while penalizing absorption and price-flo
# Stats: {"live_returns": [-1.0, -0.7290986776351929, -0.08022461086511612, -0.8899125456809998, -0.2813928723335266, -0.06580036878585815, -0.5617086291313171, -0.10828935354948044, -0.05505957826972008, -1.0, -0.5390967726707458, -0.5081847310066223, -0.43009188771247864, -0.08568576723337173, -0.05439654365181923, 0.4290037751197815, -1.0, 4.245841, -0.06407245248556137, 0.24779486656188965, -0.08369424939155579, -0.6652962565422058, -0.3431927561759949, 0.5564215183258057, -0.107005774974823, -0.6783567070960999, -0.3769824802875519, -1.0, -0.45680245757102966, -0.40098097920417786, 0.007295161485671997, 1.1187243461608887, -1.0, -0.2797567844390869, 3.1194439999999997, -0.30054157972335815, -0.7851195931434631, -0.10406997799873352, -0.47662320733070374, -0.18528439104557037, -0.876594066619873, 2.39541, 0.367870569229126, -0.13138431310653687, -0.29802918434143066, 0.09908425062894821, 0.1870144009590149, -0.26657482981681824, -0.06547584384679794, -1.0, -0.10636938363313675, -1.0, -0.9831086993217468, 0.16493013501167297, -0.2983052730560303, -0.10273964703083038, -0.5912058353424072, -0.727156937122345, -1.0, -1.0, -1.0], "live_pnl": -11.204833030700684, "live_wins": 12, "live_total": 61, "age": 90, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.2518294808665276, "backtest_acc": 0.5416359616801769, "backtest_n": 2154}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    synergy between aggressive flow (Delta) and book pressure (OBI).
    """
    df = df.copy()

    # 1. Delta-OBI Synergy: Do aggressive trades align with the book imbalance?
    # Positive = Aggressive buying + positive OBI (Sustainable move)
    # Negative = Aggressive buying + negative OBI (Absorption/Wall)
    df['delta_obi_prod'] = df['mean_delta'] * df['mean_obi']
    
    # 2. Delta-OBI Intensity Ratio: Is the move driven by flow or just a thin book?
    df['delta_obi_ratio'] = df['mean_delta'] / (df['mean_obi'].abs() + 1e-6)

    # 3. Temporal Velocity: Capturing the acceleration of price and flow
    # We group by market_slug to ensure we don't leak data between windows
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 4. Cross-Exchange Delta Dispersion: Is the flow consensus or fragmented?
    # High dispersion means exchanges are seeing different aggressive flows.
    df['delta_dispersion'] = df[['binance_delta', 'coinbase_delta', 'kraken_delta']].std(axis=1)

    # Drop metadata but keep pm_costs and other features for the model
    # Note: seconds_remaining is kept as it is a crucial feature for the model to learn time-decay
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters optimized for interaction features."""
    return {
        'n_estimators': 1500,
        'max_depth': 3,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Synergistic Flow-Book Alignment".
    The strategy seeks moves where aggressive market orders (Delta) are 
    supported by the limit order book (OBI) in the same direction.
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

    # 2. Extract Engineered Metrics
    delta_obi_prod = features_row.get('delta_obi_prod', 0.0)
    delta_obi_ratio = features_row.get('delta_obi_ratio', 0.0)
    price_velocity = features_row.get('price_velocity', 0.0)
    delta_dispersion = features_row.get('delta_dispersion', 0.0)
    
    # 3. Dynamic Hurdle
    # Base hurdle of 3%
    base_hurdle = 0.03
    hurdle = base_hurdle

    # 4. Multiplier Logic (The "Synergy" Filter)
    multiplier = 1.0

    # A) Absorption Guard: If Delta and OBI are fighting, it's likely absorption.
    # (e.g., High Delta but negative OBI means buying into a wall)
    if delta_obi_prod < 0:
        multiplier *= 0.15
    
    # B) Flow-Book Ratio: If the ratio is extreme, the move might be too thin/unstable.
    if abs(delta_obi_ratio) > 5.0:
        multiplier *= 0.5

    # C) Exchange Consensus: If delta dispersion is high, the signal is noisy.
    if delta_dispersion > (abs(features_row.get('mean_delta', 0.0)) * 2.0):
        multiplier *= 0.6

    # D) Price-Flow Alignment: Is the price actually moving in our predicted direction?
    if side == 'Up' and price_velocity < 0:
        multiplier *= 0.4
    elif side == 'Down' and price_velocity > 0:
        multiplier *= 0.4

    # 5. Execution & Sizing
    if best_ev < hurdle:
        return None, 0.0

    # Time-decay: Reduce exposure as the window closes to avoid last-second volatility noise.
    # We use a linear decay for the final 60 seconds.
    time_factor = (seconds_remaining / 300.0)
    if seconds_remaining < 60:
        time_factor = (seconds_remaining / 60.0) * 0.5

    # Final bet fraction: Scale by edge strength and cleanliness multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction