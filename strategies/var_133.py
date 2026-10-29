# Var 133 | BT: -0.138 | Live PnL: $-17.00 | Trades: 61
# Desc: Implements "Tri-Factor Sync & Time-Decay Hurdle (TS-TDH)," which uses the directional alignment of Delta, OBI, and Momentum to dynamically adjust the required EV threshold and bet size, while increasi
# Stats: {"live_returns": [-1.0, -1.0, 0.15118699999999996, -0.1, -1.0, -1.0, -1.0, 0.12663000000000002, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -0.8611947298049927, -0.1, -1.0, 2.39541, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, -1.0, 3.727685, -1.0, 1.3396370000000002, -1.0, -1.0, 2.39541, -1.0, -0.9960806369781494, -1.0, -1.0, -1.0, -1.0, -0.1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -16.995823860168457, "live_wins": 15, "live_total": 61, "age": 112, "rounds_underwater": 21, "abstain_rounds": 0, "backtest_score": -0.1375275319891638, "backtest_acc": 0.5493241880169457, "backtest_n": 3239}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronization between aggressive flow, order book state, and momentum.
    """
    df = df.copy()

    # 1. Synchronization Features (The core of the TS-TDH strategy)
    # How well do Delta and OBI align? (1.0 if same sign, -1.0 if opposite, 0.0 if neutral)
    sign_delta = np.sign(df['sum_delta'])
    sign_obi = np.sign(df['mean_obi'])
    sign_mom = np.sign(df['mean_momentum'])
    
    df['delta_obi_sync'] = sign_delta * sign_obi
    df['mom_delta_sync'] = sign_mom * sign_delta
    
    # Total Sync: Average alignment across the three microstructure pillars
    # Range: [-1.0, 1.0]
    df['total_sync'] = (df['delta_obi_sync'] + df['mom_delta_sync']) / 2.0

    # 2. Pressure & Intensity
    # Combined force of flow and book imbalance
    df['pressure_index'] = df['sum_delta'] * df['mean_obi']
    
    # Flow intensity relative to volatility (how much effort to move price)
    df['vol_norm_delta'] = df['sum_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price Leadership & Context
    df['binance_lead'] = df['binance_price'] - (df['coinbase_price'] + df['kraken_price']) / 2.0
    df['price_velocity'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Factor Sync & Time-Decay Hurdle (TS-TDH)".

    Logic:
    1. Calculate raw EV for both sides.
    2. Calculate a 'Dynamic Hurdle':
       - Base hurdle is 3%.
       - If microstructure signals (Delta, OBI, Mom) are unaligned, the hurdle increases significantly.
       - As 'seconds_remaining' decreases, the hurdle increases to account for signal decay.
    3. Bet Sizing:
       - Scale bet size by the 'total_sync' factor. 
       - Only trade if the raw EV exceeds the dynamic hurdle.
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
    sync = features_row.get('total_sync', 0.0)
    
    # 1. Time Decay Factor (0.0 at 0s remaining, 1.0 at 300s remaining)
    # We use this to increase the hurdle as the window closes.
    time_factor = min(1.0, seconds_remaining / 300.0)

    # 2. Dynamic Hurdle Calculation
    # Base: 3%
    # Penalty for lack of sync: Up to 4% (if sync is -1, hurdle += 0.04)
    # Penalty for time decay: Up to 2% (if time_factor is 0, hurdle += 0.02)
    # Max hurdle: ~9% | Min hurdle: ~3%
    hurdle = 0.03 + (0.04 * (1.0 - sync)) + (0.02 * (1.0 - time_factor))

    if best_ev > hurdle:
        # 3. Dynamic Bet Sizing
        # We scale the bet fraction by how much the sync exceeds the 'neutral' state.
        # A perfect sync (1.0) allows for a full bet.
        # A neutral sync (0.0) scales the bet down.
        # A negative sync (reversal/absorption) is heavily penalized.
        sync_multiplier = (sync + 1.0) / 2.0 # Scales from 0.0 to 1.0
        
        # Target a 4% edge for a full bet, scaled by sync strength
        bet_fraction = (best_ev / 0.04) * sync_multiplier
        
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0