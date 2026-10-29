# Var 103 | BT: -1.349 | Live PnL: $-7.32 | Trades: 61
# Desc: Implements "Asymmetric Impact-Elasticity & Strike-Proximity Gating (AI-SPG)," which scales conviction by the price-impact-per-unit-of-delta (efficiency) and adjusts entry thresholds based on the conve
# Stats: {"live_returns": [0.01714453101158142, -0.36357933282852173, 2.4273319244384766, -0.31793853640556335, 0.4610172510147095, -0.44065335392951965, -1.0, -0.9693199396133423, -1.0, -1.0, 0.024416983127593994, 2.39541, -0.7086218595504761, -0.17695589363574982, -0.9448323845863342, 0.5115811824798584, -1.0, -0.5456451177597046, 3.1194439999999997, -0.44343096017837524, -1.0, -0.7493376731872559, -0.5116006135940552, -0.3537193536758423, 2.042513132095337, 0.4042447805404663, -0.3534374237060547, -0.6198833584785461, 0.9349124431610107, -0.9230049252510071, -0.3522735834121704, -0.4969792664051056, -1.0, -1.0, 1.3836312294006348, -0.4074779450893402, -0.26750558614730835, -0.8528313636779785, -0.7687749266624451, -0.8443928956985474, -1.0, -0.5691514611244202, 2.073133, -0.6994048953056335, 3.3770716190338135, -0.7753457427024841, -0.9683640003204346, 0.9723902344703674, 2.795066, -0.9639087915420532, 0.01700831949710846, 0.7420930862426758, -0.5505225658416748, -0.33170899748802185, -1.0, -0.9245108366012573, -1.0, -0.7300827503204346, -0.4298059046268463, -1.0, -0.6592133045196533], "live_pnl": -7.315805435180664, "live_wins": 17, "live_total": 61, "age": 121, "rounds_underwater": 41, "abstain_rounds": 0, "backtest_score": -1.349275464917909, "backtest_acc": 0.55679490097883, "backtest_n": 2789}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price-impact efficiency and delta-to-volatility dynamics.
    """
    df = df.copy()

    # 1. Price-Flow Velocity & Acceleration
    # How fast is price moving and how is the aggressive flow accelerating?
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_acceleration'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Impact Efficiency (The Core Alpha Feature)
    # Measures how much price moves per unit of aggressive delta.
    # Low efficiency = Deep market (high absorption, high conviction).
    # High efficiency = Thin market (low liquidity, high slippage/fragility).
    df['impact_efficiency'] = (df['price_velocity'].abs() / (df['mean_delta'].abs() + 1e-6))
    df['impact_efficiency'] = df['impact_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)

    # 3. Delta-to-Volatility Ratio
    # Is the aggressive flow significant relative to the current noise?
    df['delta_vol_ratio'] = df['mean_delta'].abs() / (df['mean_volatility'] + 1e-6)

    # 4. Order Book Consensus
    df['obi_consensus'] = 1.0 / (df['obi_spread'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on structural signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Asymmetric Impact-Elasticity & Strike-Proximity Gating".

    Logic:
    - Deep Market (Low Efficiency): High delta is being absorbed by deep liquidity. 
      This is high-conviction momentum. Lower EV threshold, higher bet size.
    - Thin Market (High Efficiency): Small delta causes large price swings. 
      This is fragile/noisy. Higher EV threshold, lower bet size.
    - Strike Proximity: As price approaches the strike, volatility/noise increases.
      Require higher EV when 'At-The-Money'.
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
    eff = features_row.get('impact_efficiency', 0.0)
    delta = abs(features_row.get('mean_delta', 0.0))
    prox = abs(features_row.get('price_vs_strike_pct', 0.0))
    vol = features_row.get('mean_volatility', 0.0)
    
    # 1. Determine Market Regime based on Impact Efficiency
    # Thresholds are heuristic: efficiency is price_move / delta.
    is_deep_market = eff < 0.005 
    is_thin_market = eff > 0.05

    # 2. Set Regime-Specific Requirements
    # Baseline requirement
    req_ev = 0.035
    multiplier = 1.0

    if is_deep_market and delta > 0:
        # High conviction: Market is absorbing flow efficiently
        req_ev = 0.025
        multiplier = 1.2
    elif is_thin_market and delta > 0:
        # High risk: Market is fragile/slippage-prone
        req_ev = 0.050
        multiplier = 0.5
    
    # 3. Strike Proximity Convexity Adjustment
    # Near the strike (prox < 0.1%), the market is much noisier.
    if prox < 0.001:
        req_ev += 0.015
        multiplier *= 0.8

    # 4. Time Decay
    # Reduce exposure as the window closes to avoid end-of-period noise.
    time_decay = min(1.0, seconds_remaining / 300.0)

    # 5. Execution
    if best_ev > req_ev:
        # Scale bet size by the strength of the edge relative to a 5% target
        # and apply the regime multiplier and time decay.
        bet_fraction = (best_ev / 0.05) * multiplier * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0