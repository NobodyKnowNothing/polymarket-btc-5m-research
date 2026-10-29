# Var 118 | BT: -0.509 | Live PnL: $-1.87 | Trades: 61
# Desc: Implements "Microstructure Cohesion & Impact Gating (MCIG)," which dynamically adjusts the required expected value (EV) hurdle based on whether the market is in a consensus, breakout, or absorption re
# Stats: {"live_returns": [-1.0, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, 0.08073200000000003, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 0.06987699999999997, -1.0, 2.950496, 0.02403116226196289, 1.4562249999999999, -1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -0.8705628514289856, -1.0, 0.038675000000000015, 2.39541, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, 1.890848, 2.651597, -1.0, -1.0], "live_pnl": -1.8654680252075195, "live_wins": 20, "live_total": 61, "age": 138, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.5087420573466845, "backtest_acc": 0.5494054054054054, "backtest_n": 3003}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding cohesion 
    and impact metrics to distinguish between clean breakouts and noisy absorption.
    """
    df = df.copy()

    # 1. Cohesion Index: Measures how closely the exchanges are trading together.
    # High cohesion = low price/OBI spread (efficient/consensus).
    # Low cohesion = high spread (dislocated/arbitrage/noise).
    df['cohesion_index'] = 1.0 / (df['price_spread'] + df['obi_spread'] + 1e-6)

    # 2. Impact Metric: Measures the "force" of the order flow relative to price volatility.
    # High impact = aggressive delta moving price efficiently (breakout/slippage).
    # Low impact = aggressive delta struggling against volatility (absorption/noise).
    df['flow_impact'] = df['sum_delta'].abs() / (df['price_std'] + 1e-6)

    # 3. Funding-Delta Synergy: Captures if funding and aggressive flow are aligned.
    df['funding_delta_synergy'] = df['mean_funding'] * df['sum_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Cohesion & Impact Gating (MCIG)".

    Logic:
    - The market exists in three primary regimes:
        1. Consensus: High cohesion (exchanges agree). We trade with a low hurdle.
        2. Breakout: Low cohesion but high impact (dislocation + efficient movement). 
           We trade with a low hurdle to catch the momentum.
        3. Absorption: Low cohesion and low impact (dislocation + fighting flow). 
           We raise the hurdle significantly to avoid being caught in the noise.
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
    cohesion = features_row.get('cohesion_index', 0.0)
    impact = features_row.get('flow_impact', 0.0)

    # 1. Determine the Regime-Based Hurdle
    # Default hurdle for uncertain/messy markets
    hurdle = 0.05 

    if cohesion > 1.0:
        # Regime: Consensus (Exchanges are in tight agreement)
        hurdle = 0.025
    elif impact > 2.0:
        # Regime: Breakout (Dislocation is being resolved by high-impact flow)
        hurdle = 0.025
    else:
        # Regime: Absorption/Noise (Dislocation with low impact/high friction)
        hurdle = 0.07

    # 2. Time-Decay Adjustment
    # As we approach the 5-minute mark, increase the hurdle to avoid "whipsaw" 
    # volatility in the final seconds.
    time_factor = 1.0 + (0.05 * (1.0 - (min(seconds_remaining, 300.0) / 300.0)))
    effective_hurdle = hurdle * time_factor

    # 3. Execution
    if best_ev > effective_hurdle:
        # Scale bet size: more conviction if the edge is significantly above the hurdle
        # and if we are in a high-impact/high-cohesion regime.
        confidence_multiplier = 1.0
        if cohesion > 1.0 or impact > 2.0:
            confidence_multiplier = 1.2
            
        bet_fraction = (best_ev / 0.04) * confidence_multiplier
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0