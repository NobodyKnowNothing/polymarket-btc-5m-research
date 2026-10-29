# Var 127 | BT: 0.055 | Live PnL: $-0.06 | Trades: 96
# Desc: Implements "Cross-Venue Convergence Gating (CVCG)," which scales expected value by the product of directional consensus (Delta and OBI unanimity) and price agreement (inverse of cross-exchange spread)
# Stats: {"live_returns": [2.795066, 3.1194439999999997, 1.3087718486785889, -0.9673593640327454, -1.0, 1.9791409999999998, -1.0, -0.9209176898002625, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, 2.280573, -0.850386917591095, -1.0, 1.6554160000000002, -0.8471708297729492, -1.0, -1.0, 2.073133, 2.280573, -1.0, -0.9490033984184265, -1.0, 0.9626157283782959, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -0.9411599636077881, -1.0, -1.0, 1.5854279999999998, -1.0, 1.1846420000000002, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, -0.8963094353675842, 1.0440746545791626, 2.39541, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, 1.271080493927002, 1.4562249999999999, 1.890848, 1.4471871852874756, -1.0, 3.727685, -1.0, 1.1778976917266846, -1.0, -1.0, -1.0, 2.39541, -0.9215351343154907, -1.0, -0.7598556876182556, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.1406786441802979, -1.0, -0.8814123868942261, 1.8077519999999998, -1.0, 1.7294070000000001, -0.8552236557006836, -1.0, -1.0], "live_pnl": -0.05948638916015625, "live_wins": 32, "live_total": 96, "age": 209, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.055381043944048414, "backtest_acc": 0.5783305921052632, "backtest_n": 2646}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    directional unanimity and cross-exchange price stability.
    """
    df = df.copy()

    # 1. Directional Consensus: Unanimity of aggressive flow and order book imbalance.
    # Range: [-3, 3] for each. 3 = all agree, 0 = neutral/disagreement.
    df['delta_consensus'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    )
    df['obi_consensus'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    )

    # 2. Price Agreement Ratio: How tightly coupled are the exchange prices?
    # A higher ratio means more dispersion/uncertainty.
    df['price_spread_ratio'] = df['price_spread'] / (df['binance_price'] + 1e-9)

    # 3. Price Velocity: Speed of price movement relative to the strike.
    df['price_velocity'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1.0)

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
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Venue Convergence Gating (CVCG)".

    Logic:
    - Calculate raw EV for both sides.
    - Consensus Factor: The average normalized unanimity of Delta and OBI (0.0 to 1.0).
    - Agreement Factor: Penalizes the EV if cross-exchange price spread is high.
    - Effective EV: best_ev * consensus_factor * agreement_factor.
    - This ensures we only bet when the model sees edge AND the market shows 
      unanimous directional intent AND price stability.
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
    delta_cons = features_row.get('delta_consensus', 0.0)
    obi_cons = features_row.get('obi_consensus', 0.0)
    spread_ratio = features_row.get('price_spread_ratio', 0.0)

    # 1. Consensus Factor (0.0 to 1.0)
    # We average the normalized absolute consensus of Delta and OBI.
    consensus_factor = (abs(delta_cons) / 3.0 + abs(obi_cons) / 3.0) / 2.0

    # 2. Agreement Factor (0.0 to 1.0)
    # Uses a decay function to penalize price dispersion.
    # A 1% spread (0.01) reduces agreement to ~0.9. A 5% spread reduces it to ~0.66.
    agreement_factor = 1.0 / (1.0 + (spread_ratio * 10.0))

    # 3. Calculate Effective EV
    effective_ev = best_ev * consensus_factor * agreement_factor

    # 4. Dynamic Thresholding
    # As time runs out, the cost of being wrong in a volatile window increases.
    base_threshold = 0.03
    if seconds_remaining < 60:
        base_threshold = 0.05

    if effective_ev > base_threshold:
        # Scale bet fraction linearly with effective edge, targeting 4% for full size.
        bet_fraction = effective_ev / 0.04
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0