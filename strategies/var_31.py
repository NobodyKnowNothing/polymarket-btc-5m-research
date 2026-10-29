# Var 31 | BT: 0.049 | Live PnL: $-1.42 | Trades: 118
# Desc: Lead-Lag Ignition Capture (LLIC) — Targets high-conviction moves by requiring both directional "ignition" (alignment of aggressive flow and order book imbalance) and "lead-lag" momentum (Binance price
# Stats: {"live_returns": [-0.5483490824699402, 3.9731249999999996, 1.7294070000000001, 2.39541, -0.4456450343132019, 1.7219343185424805, -1.0, 0.04885600000000001, 0.06799131631851196, 1.2613489627838135, -1.0, -1.0, 3.727685, 0.03634071350097656, 0.20957952737808228, 2.280573, 0.038675000000000015, -1.0, -1.0, -1.0, 0.8601346015930176, 2.795066, 1.0903115272521973, 0.1173022985458374, 1.890848, -0.8825114965438843, -0.35920053720474243, 0.036281704902648926, -1.0, -1.0, 3.303756, -1.0, -1.0, 1.8077519999999998, -0.9871895909309387, -1.0, -1.0, 0.3538373112678528, 1.0600993633270264, -0.7100014686584473, -1.0, -1.0, 1.6554160000000002, 3.5026543140411377, 4.245841, -1.0, -1.0, -1.0, 0.0666305422782898, -0.8257854580879211, -1.0, 1.28551, -1.0, -0.18957091867923737, -1.0, -0.420230507850647, -1.0, -1.0, 0.11687099933624268, -1.0, -1.0, -0.9879031777381897, -1.0, -1.0, -0.9805863499641418, -0.35777220129966736, -0.5097269415855408, -1.0, 3.303756, -0.32261282205581665, -0.33129629492759705, -1.0, -1.0, 2.518758, -1.0, 1.5856571197509766, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, 0.42535024881362915, -1.0, -1.0, -1.0, 1.5757310390472412, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -0.6811232566833496, 4.550646, 3.1194439999999997, -1.0, 0.016701266169548035, -1.0, -1.0, -0.4387376010417938, -1.0, -1.0, 2.39541, 0.5490399599075317, -0.33271390199661255, -0.6668462157249451, 0.9061029553413391, -1.0, -1.0, -0.3498082756996155, -1.0, -0.49731940031051636, -1.0, -0.7863891124725342], "live_pnl": -1.42091703414917, "live_wins": 41, "live_total": 118, "age": 177, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.048729360242699526, "backtest_acc": 0.5418641390205371, "backtest_n": 1878}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the lead-lag relationship between exchanges and the 'ignition' 
    signal (congruence of delta and OBI).
    """
    df = df.copy()

    # 1. Velocity calculation (Price change per snapshot)
    # We group by market_slug to ensure diffs don't bleed between different 5-min windows
    df['binance_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['coinbase_vel'] = df.groupby('market_slug')['coinbase_price'].diff().fillna(0)
    
    # 2. Lead-Lag Gap: Does Binance move before the others?
    # Positive = Binance leading Up; Negative = Binance leading Down
    df['vel_div'] = df['binance_vel'] - df['coinbase_vel']

    # 3. Ignition Index: Measures if aggressive flow (delta) is hitting the side 
    # with the corresponding imbalance (OBI). 
    # Positive = Ignition (Flow and Book aligned); Negative = Absorption (Flow vs Book)
    df['ignite'] = df['mean_delta'] * df['mean_obi']

    # 4. Volatility-normalized Lead-Lag
    df['vel_div_norm'] = df['vel_div'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Lag Ignition Capture".
    The strategy looks for 'Ignition' (flow/book alignment) and 'Lead-Lag' 
    (Binance price leading other exchanges) to confirm high-conviction moves.
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
    vel_div = features_row.get('vel_div', 0.0)
    ignite = features_row.get('ignite', 0.0)
    vol = features_row.get('mean_volatility', 0.01)
    
    # 3. Dynamic Hurdle
    # Base hurdle 2.5%, scaled by volatility to avoid noise-trading in high-vol regimes
    hurdle = 0.025 + (0.05 * vol)

    if best_ev < hurdle:
        return None, 0.0

    # 4. Confluence Logic
    # A 'perfect' trade has:
    # - Positive ignition (flow and book aligned)
    # - Leading velocity (Binance moving in the direction of the trade)
    
    is_aligned = ignite > 0
    if side == 'Up':
        is_leading = vel_div > 0
    else:
        is_leading = vel_div < 0

    # 5. Execution & Sizing
    # We use a multiplier to scale the bet based on how much the signals agree.
    # If we have EV but no alignment/lead, we reduce exposure significantly.
    multiplier = 1.0
    
    if not is_aligned:
        multiplier *= 0.4  # Major penalty for absorption/divergence
    
    if not is_leading:
        multiplier *= 0.6  # Penalty for lagging or non-leading price action

    # Time-decay: Reduce bet size as the 5-minute window expires
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # Final bet fraction: Scale by (Edge/Hurdle) and the confluence multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage/spread
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction