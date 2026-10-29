# Var 39 | BT: -0.267 | Live PnL: $-6.17 | Trades: 61
# Desc: Implements "Funding-Driven Squeeze Capture (FDSC)," which identifies high-conviction squeeze setups by aligning model predictions with the direction of potential funding-induced liquidations, while pe
# Stats: {"live_returns": [0.0467546284198761, -0.5250580906867981, 2.518758, -1.0, 4.245841, 0.006954208016395569, -0.2151147574186325, 0.012643665075302124, 1.9791409999999998, -0.9865131974220276, -0.24298280477523804, 0.33044642210006714, -1.0, 0.020046979188919067, -0.522865355014801, -1.0, 0.8679823875427246, -1.0, -1.0, -0.735170841217041, 1.7294070000000001, -1.0, 0.05316340923309326, -1.0, -1.0, -1.0, -0.8656907677650452, -0.10496482998132706, -1.0, -0.652794599533081, 0.4301060438156128, -1.0, 4.550646, 3.1194439999999997, -1.0, -0.3569636344909668, -1.0, -0.4398856461048126, -0.40663760900497437, -0.26739898324012756, -0.41367629170417786, 2.39541, 0.018139243125915527, 0.41387996077537537, -1.0, 0.45933347940444946, 1.4562249999999999, -0.8127853870391846, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.252309113740921], "live_pnl": -6.1673479080200195, "live_wins": 20, "live_total": 61, "age": 79, "rounds_underwater": 7, "abstain_rounds": 0, "backtest_score": -0.26665818244221084, "backtest_acc": 0.5509973654497554, "backtest_n": 2113}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    interaction between aggressive flow (Delta) and the cost of carry (Funding)
    to identify squeeze-prone environments.
    """
    # Create a copy to avoid SettingWithCopy warnings
    X = df.copy()

    # 1. Funding-Delta Alignment: High values indicate trend-following; 
    # Negative values indicate potential squeeze/reversal setups.
    X['funding_delta_prod'] = X['mean_funding'] * X['mean_delta']

    # 2. Delta-OBI Sync: Measures if aggressive flow is supported by book pressure.
    X['delta_obi_prod'] = X['mean_delta'] * X['mean_obi']

    # 3. Price Impact Efficiency: How much price moves per unit of delta.
    # Low values suggest high absorption (hitting a wall).
    X['price_impact'] = X['mean_delta'] / (X['price_std'] + 1e-6)

    # 4. Volatility-Spread Ratio: Measures if cross-exchange dispersion is 
    # excessive relative to current volatility.
    X['vol_spread_ratio'] = X['mean_volatility'] / (X['price_spread'] + 1e-6)

    # 5. Momentum-Delta Sync: Directional alignment of momentum and flow.
    X['momentum_delta_prod'] = X['mean_momentum'] * X['mean_delta']

    # Drop metadata and target columns to ensure no leakage
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup for XGBoost
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters optimized for noisy 5-min windows."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 15.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Driven Squeeze Capture (FDSC)".
    The strategy seeks to amplify bets when the model predicts the direction 
    of a funding-induced squeeze (e.g., high funding + negative delta = Up squeeze).
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

    # 2. Extract Microstructure Metrics
    mean_funding = features_row.get('mean_funding', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    price_std = features_row.get('price_std', 0.0)

    # 3. Squeeze Detection Logic
    # A squeeze setup exists when funding is high and delta is moving against it.
    # Squeeze direction is the direction that forces liquidations (sign of funding).
    squeeze_dir = 1 if mean_funding > 0 else (-1 if mean_funding < 0 else 0)
    squeeze_intensity = abs(mean_funding * mean_delta)
    
    # Absorption detection: High delta with low price movement
    absorption = abs(mean_delta) / (price_std + 1e-6)

    # 4. Dynamic Decision Logic
    multiplier = 1.0
    hurdle = 0.025

    # A) Squeeze Regime: High conviction if model agrees with the squeeze direction
    if squeeze_intensity > 0.0001:
        if side == ('Up' if squeeze_dir == 1 else 'Down' if squeeze_dir == -1 else None):
            multiplier = 1.5  # Aggressively scale into squeeze-aligned trades
        else:
            multiplier = 0.0  # Abstain if betting against the squeeze direction

    # B) Normal/Absorption Regime: Scale down if the market is "heavy" (absorption)
    else:
        if absorption > 2.5:
            multiplier = 0.4  # Reduce exposure when delta is being absorbed
        else:
            multiplier = 1.0

    # 5. Execution & Sizing
    if best_ev < hurdle or multiplier == 0:
        return None, 0.0

    # Time-decay: Quadratic decay to reduce exposure as the window closes
    time_factor = (seconds_remaining / 300.0) ** 2
    
    # Final bet fraction calculation
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold for profitability
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction