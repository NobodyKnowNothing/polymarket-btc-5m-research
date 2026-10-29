# Var 126 | BT: 0.160 | Live PnL: $-0.46 | Trades: 104
# Desc: Implements "Multi-Venue Consensus Scaling (MVCS)," which scales the model's expected value by a continuous consensus factor derived from the directional agreement of all three exchanges on both Delta
# Stats: {"live_returns": [-1.0, -1.0, 4.550646, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 2.795066, 3.727685, 3.1194439999999997, -1.0, 1.9791409999999998, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, 2.280573, -1.0, -1.0, -1.0, 1.583419919013977, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9335082769393921, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, 1.5191249999999998, -0.9602305293083191, 2.280573, 1.9791409999999998, -1.0, -0.97108393907547, -0.9897258877754211, -1.0, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, 3.727685, -1.0, 2.39541, -0.9746913313865662, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 1.7686777114868164, -1.0, 1.7294070000000001, -1.0, -1.0, 0.04551428556442261, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0], "live_pnl": -0.46022796630859375, "live_wins": 33, "live_total": 104, "age": 275, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.16046970174885838, "backtest_acc": 0.5787841191066998, "backtest_n": 2485}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus and flow velocity.
    """
    df = df.copy()

    # 1. Consensus Metrics: How much do the three exchanges agree on direction?
    # Range: -3 to 3 (3 = all agree, 0 = no agreement, -3 = all opposite)
    df['delta_agreement_sum'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    )
    
    df['obi_agreement_sum'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    )

    # 2. Disagreement Magnitude: Standard deviation of signals across venues
    df['delta_std'] = df[['binance_delta', 'coinbase_delta', 'kraken_delta']].std(axis=1)
    df['obi_std'] = df[['binance_obi', 'coinbase_obi', 'kraken_obi']].std(axis=1)

    # 3. Flow Velocity: Rate of change of the aggregate delta within a market window
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to venue-specific noise."""
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
    Decide whether to trade using "Multi-Venue Consensus Scaling (MVCS)".

    Logic:
    - Calculate raw EV for both sides.
    - Calculate a 'Consensus Score' (0.0 to 1.0) based on the agreement of 
      Delta and OBI across all three exchanges.
    - Apply the Consensus Score to the EV: High disagreement (low score) 
      drastically reduces the effective edge.
    - Dynamic Thresholding: The required effective EV increases as 
      'seconds_remaining' decreases to avoid late-window noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered consensus features
    # We use abs() because agreement can be positive or negative
    delta_agreement = abs(features_row.get('delta_agreement_sum', 0.0))
    obi_agreement = abs(features_row.get('obi_agreement_sum', 0.0))
    
    # Consensus Score: Average of normalized agreement (0 to 1)
    # If all 3 agree, score is (3/3 + 3/3)/2 = 1.0. 
    # If 2 agree, score is (1/3 + 1/3)/2 = 0.33.
    consensus_score = ( (delta_agreement / 3.0) + (obi_agreement / 3.0) ) / 2.0

    # 1. Calculate Effective EV
    effective_ev = best_ev * consensus_score

    # 2. Dynamic Thresholding
    # Base threshold is 0.03. As time runs out (approaches 0), 
    # the threshold scales up to 0.06 to demand higher conviction.
    base_threshold = 0.03
    time_decay_factor = 1.0 + (300 - seconds_remaining) / 300
    dynamic_threshold = base_threshold * time_decay_factor

    if effective_ev > dynamic_threshold:
        # Scale bet fraction based on how much effective EV exceeds the threshold
        # We aim for a full bet when effective_ev is ~5%
        bet_fraction = effective_ev / 0.05
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0