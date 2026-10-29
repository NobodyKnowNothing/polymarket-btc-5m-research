# Var 47 | BT: 1.011 | Live PnL: $-13.64 | Trades: 61
# Desc: Implements "Signal-to-Noise Ratio (SNR) Adaptive Scaling," which treats the model's Expected Value as a function of volatility-normalized order flow, scaling bets higher when aggressive delta is large
# Stats: {"live_returns": [1.28551, -0.7035161256790161, 4.245841, -1.0, -0.9696080088615417, -0.49799948930740356, -1.0, -0.6011992692947388, -1.0, -1.0, -0.712722659111023, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -0.7336978316307068, -0.9663611054420471, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -0.6006593108177185, -1.0, -0.730655312538147, -1.0, 2.950496, 1.1323907375335693, 3.9731249999999996, -1.0, -1.0, 2.280573, 1.8077519999999998, -1.0, -1.0, 4.893558, -1.0, -0.8680251240730286, -1.0, -0.7148988246917725, -1.0, 2.651597, -1.0, 0.07705330848693848, -0.9441922307014465, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.903401255607605, -1.0, -1.0, -1.0, -1.0, -1.0, -0.897975742816925], "live_pnl": -13.641629219055176, "live_wins": 13, "live_total": 61, "age": 205, "rounds_underwater": 21, "abstain_rounds": 0, "backtest_score": 1.0107383208822047, "backtest_acc": 0.5564899451553931, "backtest_n": 661}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Signal-to-Noise Ratios (SNR) by normalizing flow and book pressure 
    against the current volatility regime.
    """
    df = df.copy()

    # 1. Signal-to-Noise Ratios (SNR): Normalize flow and OBI by volatility
    # This helps the model understand if a delta move is 'significant' or just noise.
    vol_buffer = df['mean_volatility'] + 1e-6
    df['delta_snr'] = df['mean_delta'] / vol_buffer
    df['obi_snr'] = df['mean_obi'] / vol_buffer
    df['funding_snr'] = df['mean_funding'] / vol_buffer
    
    # 2. Spread-to-Volatility: Is the cross-exchange dispersion high relative to vol?
    df['spread_snr'] = df['price_spread'] / vol_buffer

    # 3. Flow-Book Congruence: Product of delta and OBI (directional strength)
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 4. Temporal Velocity: Rate of change in delta and OBI within the market window
    # Grouping by market_slug to prevent cross-market leakage
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 5. Funding-Delta Divergence: High funding + opposite delta = potential squeeze
    df['funding_delta_divergence'] = df['mean_funding'] * df['mean_delta']

    # Drop metadata but keep pm_costs and other features for the model
    # We keep pm_cost_up/down as they are crucial for the model to learn mispricing
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
    Decide whether to trade using "Signal-to-Noise Ratio (SNR) Adaptive Scaling".
    The strategy scales the bet size based on the magnitude of the 'clean' 
    directional signal (Delta/Vol) relative to the expected value.
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

    # 2. Extract Engineered SNR Metrics
    delta_snr = abs(features_row.get('delta_snr', 0.0))
    obi_snr = abs(features_row.get('obi_snr', 0.0))
    spread_snr = features_row.get('spread_snr', 0.0)
    funding_delta_div = features_row.get('funding_delta_divergence', 0.0)

    # 3. Dynamic Hurdle Calculation
    # We increase the required EV if the exchange spread is wide (high uncertainty)
    # or if the funding/delta divergence is extreme (potential volatility spike).
    base_hurdle = 0.025
    uncertainty_penalty = (spread_snr * 0.005) + (abs(funding_delta_div) * 0.01)
    hurdle = base_hurdle + uncertainty_penalty

    if best_ev < hurdle:
        return None, 0.0

    # 4. Signal Strength (SNR) Calculation
    # We combine Delta SNR and OBI SNR to find 'high-conviction' microstructure states.
    # We use a geometric mean approach to ensure both signals must be present.
    signal_strength = math.sqrt(delta_snr * obi_snr) if (delta_snr > 0 and obi_snr > 0) else (delta_snr + obi_snr)

    # 5. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid last-second volatility.
    time_factor = (seconds_remaining / 300.0)
    
    # Bet fraction is scaled by:
    # (Edge / Hurdle) -> Higher edge = higher bet
    # (Signal Strength + 1) -> Higher SNR = higher bet
    # (Time Factor) -> More time remaining = higher bet
    bet_fraction = (best_ev / hurdle) * (signal_strength + 1.0) * time_factor
    
    # Cap bet_fraction to manage risk
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold for execution
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction