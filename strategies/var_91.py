# Var 91 | BT: -0.947 | Live PnL: $-16.24 | Trades: 61
# Desc: Scales expected value by the product of directional coherence (OBI vs Delta) and cross-exchange agreement (OBI spread), while dynamically raising the entry hurdle during high-volatility or late-window
# Stats: {"live_returns": [1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8157181739807129, -1.0, -1.0, 0.06987699999999997, 0.06987699999999997, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, 0.16389999999999993, 1.8077519999999998, 2.173394, -1.0, 1.1846420000000002, -1.0, 4.550646, 0.036664485931396484, 3.1194439999999997, -1.0, 0.09182599999999996, -1.0, -1.0, -0.7657223343849182, -0.5551276803016663, -1.0, 2.39541, 1.0291078090667725, -1.0, 1.4049733877182007, -1.0, -0.7188399434089661, -1.0, -0.6508749127388, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8079806566238403, -1.0, 2.073133, 0.0551983118057251, -1.0, -1.0], "live_pnl": -16.238338470458984, "live_wins": 18, "live_total": 61, "age": 98, "rounds_underwater": 59, "abstain_rounds": 0, "backtest_score": -0.946729911515932, "backtest_acc": 0.5691194209891436, "backtest_n": 3191}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    directional coherence and cross-exchange consensus.
    """
    df = df.copy()

    # 1. OBI-Delta Coherence: Do the book imbalance and aggressive flow point the same way?
    # Range: [-1, 1]. 1.0 = perfect alignment, -1.0 = perfect opposition.
    df['obi_delta_coherence'] = df['mean_obi'] * np.sign(df['mean_delta'])

    # 2. OBI Consensus: How much do the exchanges agree on the order book state?
    # We use the inverse of the spread. 1.0 = perfect agreement, 0.0 = max disagreement.
    df['obi_consensus'] = (1.0 - df['obi_spread']).clip(0, 1)

    # 3. Volatility-Normalized Delta: How much flow is moving the market relative to noise?
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Velocity: Rate of price change relative to volatility
    df['price_change_pct'] = df.groupby('market_slug')['binance_price'].pct_change()
    df['price_velocity'] = df['price_change_pct'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Coherence-Consensus Scaling (CCS)".

    Logic:
    - Calculate raw EV for both sides.
    - Calculate a 'Coherence Multiplier' based on OBI and Delta alignment.
    - Calculate a 'Consensus Multiplier' based on cross-exchange OBI agreement.
    - Apply a dynamic threshold that increases with volatility and as time runs out.
    - Scale bet size based on the 'cleanliness' of the signal (multiplier).
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
    coherence = features_row.get('obi_delta_coherence', 0.0)
    consensus = features_row.get('obi_consensus', 0.0)
    vol = features_row.get('mean_volatility', 0.01)

    # 1. Multiplier: Combines Coherence (0 to 2) and Consensus (1 to 2)
    # If coherence is -1 (opposition), multiplier becomes 0.
    # If coherence is 1 and consensus is 1, multiplier is 4.
    multiplier = (1.0 + coherence) * (1.0 + consensus)

    # 2. Effective EV
    effective_ev = best_ev * multiplier

    # 3. Dynamic Thresholding
    # Base threshold is 3%. We add penalties for high volatility and low time remaining.
    # vol_factor scales the penalty based on current volatility regime.
    vol_factor = min(1.0, vol / 0.02) 
    # time_factor increases as seconds_remaining approaches 0.
    time_factor = 1.0 - (min(300.0, seconds_remaining) / 300.0)
    
    threshold = 0.03 + (0.02 * vol_factor) + (0.02 * time_factor)

    if effective_ev > threshold:
        # 4. Dynamic Bet Sizing
        # We scale the bet fraction by the multiplier. 
        # A 'clean' trade (multiplier=4) gets much higher conviction than a 'noisy' one.
        # We use 0.04 as a normalization constant for the base EV.
        bet_fraction = (best_ev / 0.04) * (multiplier / 2.0)
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0