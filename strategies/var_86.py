# Var 86 | BT: 0.053 | Live PnL: $-9.97 | Trades: 61
# Desc: Implements "Contrarian Absorption & Strike-Proximity Scaling (CASPS)," which targets high-alpha reversal opportunities by identifying when aggressive flow (Delta) is being absorbed by the order book (
# Stats: {"live_returns": [-1.0, 1.1375739999999999, -1.0, 2.39541, -1.0, 6.836794, 2.950496, 3.505625, 2.173394, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 3.9731249999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.092555, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0], "live_pnl": -9.967368, "live_wins": 13, "live_total": 61, "age": 106, "rounds_underwater": 15, "abstain_rounds": 0, "backtest_score": 0.052710050285948815, "backtest_acc": 0.5435643564356436, "backtest_n": 2455}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure divergence (absorption) and flow velocity.
    """
    df = df.copy()

    # 1. Velocity Features: Capture the rate of change in microstructure
    # We group by market_slug to ensure diffs don't leak between different markets
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Divergence/Ratio Features: Capture the mismatch between intent (OBI) and action (Delta)
    # A negative value indicates 'Absorption' (e.g., Delta is positive, but OBI is negative)
    df['delta_obi_divergence'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])
    df['delta_obi_ratio'] = df['mean_delta'] / (df['mean_obi'] + 1e-6)

    # Drop metadata and target columns
    # We drop everything that isn't a predictive feature for the XGBoost model
    cols_to_drop = [
        'market_slug', 'timestamp', 'actual_outcome', 
        'seconds_remaining', 'pm_cost_up', 'pm_cost_down'
    ]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle potential infinities from division and NaNs from diffs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Contrarian Absorption & Strike-Proximity Scaling (CASPS)".

    Logic:
    1. Calculate raw EV for both sides.
    2. Detect 'Absorption Divergence': If Delta and OBI have opposite signs, the 
       market is likely absorbing aggressive flow. We flip the model's side to 
       bet on the reversal (Contrarian).
    3. Apply 'Strike Proximity Scaling': The edge is most exploitable when the 
       price is near the strike. We scale down the effective EV if the price 
       is far from the strike.
    4. Apply 'Dispersion Penalty': Scale down if exchange prices are diverging.
    5. Dynamic Threshold: Increase the required EV as the market window closes.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract features
    delta = features_row.get('mean_delta', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    divergence = np.sign(delta) * np.sign(obi)
    dispersion_pct = features_row.get('price_dispersion_pct', 0.0)
    strike_dist_pct = abs(features_row.get('price_vs_strike_pct', 0.0))

    # 1. Determine Base Side and EV
    if ev_up > ev_down:
        side = 'Up'
        base_ev = ev_up
    else:
        side = 'Down'
        base_ev = ev_down

    # 2. Contrarian Absorption Logic
    # If divergence < 0, Delta and OBI are fighting. 
    # If Delta is positive (buying) but OBI is negative (selling pressure), bet DOWN.
    effective_ev = base_ev
    if divergence < 0:
        if delta > 0: # Aggressive buying vs negative OBI -> Reversal Down
            side = 'Down'
            effective_ev = ev_down * 1.35
        elif delta < 0: # Aggressive selling vs positive OBI -> Reversal Up
            side = 'Up'
            effective_ev = ev_up * 1.35

    # 3. Scaling Factors
    # Proximity Factor: High when price is near strike, low when far.
    # We use a decay function: 1 / (1 + distance)
    proximity_factor = 1.0 / (1.0 + (strike_dist_pct * 15.0))
    
    # Dispersion Penalty: Reduces conviction when exchanges disagree.
    dispersion_penalty = 1.0 / (1.0 + (dispersion_pct * 100.0))

    # 4. Final Effective EV
    final_ev = effective_ev * proximity_factor * dispersion_penalty

    # 5. Time-Gated Thresholding
    # As seconds_remaining -> 0, we require a higher EV to account for volatility decay.
    # Threshold moves from ~0.03 at start to ~0.07 at end.
    base_threshold = 0.03 + (1.0 - (seconds_remaining / 300.0)) * 0.04

    if final_ev > base_threshold:
        # Scale bet fraction: target a 5% edge for a full 1.0 bet.
        bet_fraction = final_ev / 0.05
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0