# Var 89 | BT: -0.331 | Live PnL: $-0.79 | Trades: 62
# Desc: Implements "Multi-Exchange Flow Consensus (MEFC)," which scales conviction and entry thresholds based on the degree of agreement between aggressive trade flows across Binance, Coinbase, and Kraken, ra
# Stats: {"live_returns": [3.9731249999999996, 2.950496, -0.6795408725738525, -1.0, -0.9336824417114258, -1.0, -0.9462274312973022, -0.8587523698806763, -1.0, -1.0, 1.8077519999999998, -0.9532675743103027, -0.6991260051727295, -1.0, -0.9532675743103027, -1.0, -1.0, -1.0, -0.1, -1.0, -1.0, 2.651597, 7.543445, 1.5854279999999998, -0.8221542239189148, -0.1, -1.0, -1.0, -0.784704864025116, -0.1, -1.0, -1.0, -0.1, 0.3119444, -1.0, -1.0, -0.8491955995559692, 2.39541, -1.0, 1.105822205543518, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -0.6748237013816833, -0.8382771015167236, -1.0, -0.1, -1.0, -0.7102925777435303, 2.073133, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, 1.7294070000000001, -0.7014795541763306, -0.1], "live_pnl": -0.7896469593048095, "live_wins": 15, "live_total": 62, "age": 135, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.33143052627428726, "backtest_acc": 0.577461330714461, "backtest_n": 2459}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronicity of aggressive flow across the three major exchanges.
    """
    df = df.copy()

    # 1. Flow Consensus: How many exchanges agree on the direction of aggressive flow?
    # 1.0 = All three agree, 0.33 = Two agree, 0.0 = No agreement or neutral.
    s1 = np.sign(df['binance_delta'])
    s2 = np.sign(df['coinbase_delta'])
    s3 = np.sign(df['kraken_delta'])
    
    # Count pairwise agreements
    agreement_count = (s1 == s2).astype(int) + (s2 == s3).astype(int) + (s1 == s3).astype(int)
    df['flow_consensus'] = agreement_count / 3.0

    # 2. Delta Spread: The absolute divergence in aggressive flow magnitude.
    # High spread indicates one exchange is experiencing much higher activity/imbalance than others.
    df['delta_spread'] = abs(df['binance_delta'] - df['coinbase_delta']) + \
                         abs(df['coinbase_delta'] - df['kraken_delta'])

    # 3. OBI-Delta Divergence: Do the limit orders (OBI) align with the aggressive flow (Delta)?
    # 1.0 = Perfect alignment, -1.0 = Total opposition.
    df['obi_delta_sync'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

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
    Decide whether to trade using "Multi-Exchange Flow Consensus (MEFC)".

    Logic:
    - Calculate raw EV for both sides.
    - Use 'flow_consensus' to dynamically adjust the required EV threshold.
    - If exchanges disagree on flow (low consensus), the signal is noisy; 
      we require a much higher EV to enter.
    - Scale the bet size (bet_fraction) proportionally to the consensus level.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract consensus feature
    flow_consensus = features_row.get('flow_consensus', 0.5)

    # 1. Dynamic Thresholding
    # Base threshold is 0.03. If consensus is 0.0 (no agreement), threshold rises to 0.10.
    # If consensus is 1.0 (perfect agreement), threshold stays at 0.03.
    base_threshold = 0.03
    max_threshold_penalty = 0.07
    dynamic_threshold = base_threshold + (max_threshold_penalty * (1.0 - flow_consensus))

    # 2. Time-based tightening
    # As the window closes, we increase the threshold to avoid high-volatility noise.
    if seconds_remaining < 60:
        dynamic_threshold *= 1.5
    elif seconds_remaining < 120:
        dynamic_threshold *= 1.2

    # 3. Execution Decision
    if best_ev > dynamic_threshold:
        # Scale bet fraction by both the edge magnitude and the consensus strength.
        # We target a 4% edge for a full bet in high-consensus environments.
        edge_scaling = best_ev / 0.04
        consensus_scaling = flow_consensus
        
        bet_fraction = edge_scaling * consensus_scaling
        
        # Clamp bet_fraction between 0.1 and 1.0
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0