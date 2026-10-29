# Var 104 | BT: -0.298 | Live PnL: $-10.65 | Trades: 61
# Desc: Implements "Cross-Venue Dispersion & Funding-Flow Tension Gating (CV-FTG)," which targets alpha by detecting arbitrage convergence (high spread + velocity) and crowded-trade exhaustion (high impact-co
# Stats: {"live_returns": [0.04714745283126831, 0.06365585327148438, -0.8848223090171814, 2.795066, -0.9737321734428406, 1.3964720000000002, -0.4790336787700653, -0.49995142221450806, -1.0, -1.0, -1.0, 0.04676103591918945, -1.0, 2.39541, -1.0, -1.0, -0.22547291219234467, -1.0, -0.5522103905677795, -1.0, 0.12699776887893677, -1.0, 4.550646, 3.1194439999999997, -1.0, 0.047094643115997314, -1.0, -1.0, -1.0, -0.7585756778717041, -1.0, -0.8055996894836426, 2.39541, 0.8976887464523315, -0.8049576282501221, -1.0, 1.4562249999999999, -0.5225218534469604, -1.0, -0.8023070096969604, -1.0, -0.4116418659687042, -0.8928080201148987, -1.0, 1.9791409999999998, -0.9029041528701782, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4102202355861664, -0.2758089005947113, -0.5042410492897034, 2.073133, 0.034504443407058716, -1.0, 3.505625, -0.8771325349807739], "live_pnl": -10.653519630432129, "live_wins": 17, "live_total": 61, "age": 91, "rounds_underwater": 35, "abstain_rounds": 0, "backtest_score": -0.29843900625626074, "backtest_acc": 0.5665451230628988, "backtest_n": 3517}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    impact-cost (absorption), funding-flow tension, and exchange dispersion.
    """
    df = df.copy()

    # 1. Price Velocity: Change in mid-price per snapshot
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Impact Cost: How much delta is required to move the price (Absorption metric)
    # High impact_cost = High absorption (lots of delta, little price movement)
    # Low impact_cost = High efficiency (little delta, lots of price movement)
    df['impact_cost'] = df['mean_delta'].abs() / (df['price_velocity'].abs() + 1e-8)
    df['impact_cost'] = df['impact_cost'].replace([np.inf, -np.inf], 0).fillna(0).clip(upper=1000)

    # 3. Funding-Flow Tension: Alignment/Divergence between funding and aggressive flow
    # Positive = Longs paying + Buying pressure (Trend)
    # Negative = Longs paying + Selling pressure (Potential Squeeze/Exhaustion)
    df['funding_tension'] = df['mean_funding'] * df['mean_delta']

    # 4. Dispersion Velocity: How fast the exchange price spread is changing
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to micro-noise."""
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
    Decide whether to trade using "Cross-Venue Dispersion & Funding-Flow Tension Gating (CV-FTG)".

    Logic:
    - Convergence Burst: High price spread + high velocity. Arbitrage is resolving.
    - Tension Absorption: High impact_cost + high funding tension. Aggressive flow is being absorbed.
    - Efficient Momentum: Low impact_cost + high delta. Price is moving easily with flow.
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
    ic = features_row.get('impact_cost', 0.0)
    ft = features_row.get('funding_tension', 0.0)
    ps = features_row.get('price_spread', 0.0)
    pv = features_row.get('price_velocity', 0.0)
    delta = features_row.get('mean_delta', 0.0)

    # 1. Identify Microstructure Regimes
    # Convergence: Exchanges are out of sync and price is moving to bridge the gap
    is_convergence = ps > 0.0005 and abs(pv) > 0.0001
    
    # Tension: Heavy flow is hitting a wall (high absorption) and funding is extreme
    is_tension = ic > 50.0 and abs(ft) > 0.00001
    
    # Momentum: Price is moving efficiently with the flow
    is_momentum = ic < 5.0 and abs(delta) > 0.0005

    # 2. Set Regime-Specific Thresholds and Multipliers
    if is_convergence:
        # Fast arb resolution: lower threshold, moderate size
        req_ev = 0.025
        multiplier = 1.1
    elif is_tension:
        # High-conviction absorption/squeeze: high threshold, high size
        req_ev = 0.040
        multiplier = 1.3
    elif is_momentum:
        # Standard trend: moderate threshold, lower size (to avoid chasing)
        req_ev = 0.030
        multiplier = 0.8
    else:
        # Noise/Neutral
        req_ev = 0.045
        multiplier = 0.5

    # 3. Apply Time Gating
    # As the window closes, convergence plays become much more reliable
    time_decay = min(1.0, seconds_remaining / 300.0)
    if is_convergence and seconds_remaining < 120:
        req_ev *= 0.8  # Lower threshold for late-window convergence

    if best_ev > req_ev:
        # Scale bet size by edge magnitude and regime multiplier
        # Target 5% EV as the baseline for 1.0 bet_fraction
        bet_fraction = multiplier * (best_ev / 0.05) * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0