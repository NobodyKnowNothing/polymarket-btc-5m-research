# Var 35 | BT: -0.146 | Live PnL: $-6.93 | Trades: 61
# Desc: Targets "clean" breakouts by identifying moments where price movement is highly efficient (high price impact per unit of delta) and cross-exchange price spreads are compressing, filtering out "absorpt
# Stats: {"live_returns": [2.073133, -1.0, 0.13876599999999994, 1.890848, -1.0, 0.08396559953689575, -1.0, 1.6554160000000002, 3.727685, 4.245841, 0.06044560670852661, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -0.34858620166778564, -1.0, -0.7250667810440063, -1.0, -0.06934738904237747, -1.0, 0.15118699999999996, 0.032314419746398926, -1.0, -1.0, -0.33498379588127136, -0.4754994809627533, -1.0, -0.18303632736206055, -1.0, -1.0, 0.09182599999999996, -1.0, 0.13876599999999994, -0.25585034489631653, 4.550646, 3.1194439999999997, -0.5877702236175537, -1.0, -1.0, -1.0, -0.36260679364204407, -1.0, -0.9057928323745728, 2.39541, 0.015703290700912476, 0.9616411924362183, -1.0, -1.0, 0.6073680520057678, 0.7386688590049744, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.6493389010429382], "live_pnl": -6.933293342590332, "live_wins": 20, "live_total": 61, "age": 74, "rounds_underwater": 8, "abstain_rounds": 0, "backtest_score": -0.14611492023733136, "backtest_acc": 0.5561165048543689, "backtest_n": 2089}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Microstructure Efficiency (Impact) and Exchange Convergence (Compression).
    """
    df = df.copy()

    # 1. Velocity and Spread Dynamics (Grouped by market to prevent leakage)
    group = df.groupby('market_slug')
    
    # Price velocity on the primary exchange
    df['binance_vel'] = group['binance_price'].diff().fillna(0)
    
    # Cross-exchange spread velocity (Negative = Convergence/Compression)
    df['spread_vel'] = group['price_spread'].diff().fillna(0)
    
    # 2. Price Impact Efficiency: How much price moves per unit of aggressive delta
    # High impact = efficient trend; Low impact = absorption/resistance
    df['price_impact'] = (df['binance_vel'].abs() / (df['binance_delta'].abs() + 1e-6))
    df['price_impact'] = df['price_impact'].clip(0, 100) # Cap to prevent outliers

    # 3. Ignition: Alignment of aggressive flow and order book imbalance
    df['ignite'] = df['binance_delta'] * df['binance_obi']

    # 4. OBI Momentum
    df['obi_vel'] = group['mean_obi'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prioritize robust microstructure signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Exchange Compression & Impact Efficiency (CECIE)".
    Filters for moves that are both efficient (high impact) and consensus-building (compressing spread).
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
    impact = features_row.get('price_impact', 0.0)
    spread_vel = features_row.get('spread_vel', 0.0)
    ignite = features_row.get('ignite', 0.0)
    binance_vel = features_row.get('binance_vel', 0.0)

    # 3. Dynamic Hurdle
    # Base hurdle 3% to account for the complexity of microstructure signals
    hurdle = 0.03

    if best_ev < hurdle:
        return None, 0.0

    # 4. Confluence Logic: Efficiency, Compression, and Alignment
    
    # Efficiency: Is the price actually moving relative to the delta being traded?
    # We want high impact (price moving easily)
    is_efficient = impact > 0.01 
    
    # Compression: Are the exchanges converging on the new price?
    # spread_vel < 0 means the spread is narrowing (convergence)
    is_compressing = spread_vel <= 0
    
    # Alignment: Is the ignition (delta * OBI) moving in the same direction as our bet?
    is_aligned = (ignite > 0 and side == 'Up') or (ignite < 0 and side == 'Down')

    # 5. Execution & Sizing
    # We use a multiplicative approach to scale the bet based on signal quality.
    multiplier = 1.0

    # Efficiency Multiplier: Penalize absorption (low impact)
    if is_efficient:
        multiplier *= 1.3
    else:
        multiplier *= 0.4

    # Compression Multiplier: Penalize divergence (widening spread)
    if is_compressing:
        multiplier *= 1.3
    else:
        multiplier *= 0.5

    # Alignment Multiplier: Penalize flow/book disagreement
    if is_aligned:
        multiplier *= 1.3
    else:
        multiplier *= 0.4

    # Time-decay: Reduce bet size as the 5-minute window expires
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # Final bet fraction calculation
    # Scale by (Edge/Hurdle) and apply the confluence multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction