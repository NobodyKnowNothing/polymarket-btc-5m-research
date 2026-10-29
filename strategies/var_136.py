# Var 136 | BT: 0.391 | Live PnL: $-1.54 | Trades: 67
# Desc: Implements "Synthetic Price Dislocation (SPD) Gating," which targets alpha by identifying divergences between the weighted exchange mid-price and the Polymarket implied price.
# Stats: {"live_returns": [1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, -1.0, -1.0, 2.39541, 1.28551, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, 3.727685, -1.0, 1.3396370000000002, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, 1.1846420000000002, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.5412749999999997, "live_wins": 22, "live_total": 67, "age": 121, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.3910484532025591, "backtest_acc": 0.5676488274203247, "backtest_n": 3674}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    dislocation between exchange-implied price and Polymarket costs.
    """
    df = df.copy()

    # 1. Calculate Synthetic Mid-Price (Weighted by liquidity/importance)
    # Binance (50%), Coinbase (30%), Kraken (20%)
    df['synth_price'] = (df['binance_price'] * 0.5) + (df['coinbase_price'] * 0.3) + (df['kraken_price'] * 0.2)

    # 2. Percentage distance of synthetic price from strike
    df['synth_dist_strike_pct'] = (df['synth_price'] - df['strike_price']) / df['strike_price']

    # 3. Polymarket implied distance from fair (0.5)
    # If cost_up is 0.6, pm_dist_fair is 0.1
    df['pm_dist_fair'] = df['pm_cost_up'] - 0.5

    # 4. Dislocation Index
    # Compares the percentage price move to the PM probability move.
    # We scale PM distance by 0.01 to bring it to a similar magnitude as synth_dist_strike_pct.
    df['dislocation_index'] = df['synth_dist_strike_pct'] - (df['pm_dist_fair'] * 0.01)

    # 5. Dislocation Velocity (Rate of change of the dislocation)
    df['dislocation_velocity'] = df.groupby('market_slug')['dislocation_index'].diff().fillna(0)

    # 6. Volatility-Normalized Dislocation
    # Measures how many "standard deviations" the price is from strike relative to current dispersion
    df['synth_dist_strike_vol_adj'] = df['synth_dist_strike_pct'] / (df['price_std'] / df['strike_price'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.006,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Synthetic Price Dislocation (SPD) Gating".

    Logic:
    - Calculate raw EV for both sides.
    - Alignment Check: A trade is only high-conviction if the model's direction 
      aligns with the 'dislocation' (e.g., we want 'Up' if exchanges are 
      priced higher than Polymarket).
    - Alignment Multiplier: If aligned, amplify EV; if contradicting, penalize heavily.
    - Time-Decay Threshold: As seconds_remaining decreases, the required 
      effective EV threshold increases, as there is less time for dislocation to resolve.
    - Dynamic Sizing: Scale bet fraction by the magnitude of effective EV.
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
    dislocation = features_row.get('dislocation_index', 0.0)

    # 1. Alignment Factor
    # If we are betting 'Up', we want dislocation > 0 (Exchanges > PM)
    # If we are betting 'Down', we want dislocation < 0 (Exchanges < PM)
    if (side == 'Up' and dislocation > 0) or (side == 'Down' and dislocation < 0):
        alignment_multiplier = 1.5
    else:
        alignment_multiplier = 0.2  # Heavily penalize trades where PM and Exchanges disagree

    # 2. Calculate Effective EV
    effective_ev = best_ev * alignment_multiplier

    # 3. Dynamic Thresholding (Increases as time runs out)
    # At 300s, threshold is ~0.03. At 0s, threshold is ~0.08.
    time_progress = seconds_remaining / 300.0
    base_threshold = 0.03
    dynamic_threshold = base_threshold + (1.0 - time_progress) * 0.05

    if effective_ev > dynamic_threshold:
        # 4. Dynamic Bet Sizing
        # Scale bet fraction based on effective EV. 
        # An effective EV of 0.05 results in a 1.0 bet.
        bet_fraction = effective_ev / 0.05
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0