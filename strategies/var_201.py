# Var 201 | BT: 0.085 | Live PnL: $-12.54 | Trades: 64
# Desc: Implements "Cross-Venue Coherence Scaling (CVCS)," which scales bet size based on the degree of consensus between Binance, Coinbase, and Kraken's OBI and Delta to avoid "fakeout" moves driven by singl
# Stats: {"live_returns": [-1.0, -0.1, 0.03646693727941297, -0.15333770080659853, -0.1, -1.0, -0.3608176437821805, -1.0, -0.1, -0.1, 0.008885224649999968, -1.0, 2.173394, -0.1, -0.1, 2.950496, -0.1, -1.0, -0.1, -1.0, 0.2280573, -1.0, -1.0, -1.0, 0.013335430849999996, -1.0, 0.3119444, -0.1, 0.008413746249999993, -1.0, -1.0, -1.0, 0.047095561838296995, 1.9791409999999998, -1.0, -1.0, 1.890848, -0.1, 1.5191249999999998, 1.049452, -1.0, 1.233902, -1.0, -1.0, -1.0, -0.3130007210834712, 0.1092555, -1.0, -1.0, -1.0, 0.12663000000000002, -0.265575, 1.890848, -0.1, -1.0, 0.2651597, -0.11401984569865153, -0.2315751593980619, -1.0, -1.0, 0.11476700000000006, -1.0, 0.033681864053730026, 0.006101211181867627], "live_pnl": -12.541326194665656, "live_wins": 21, "live_total": 64, "age": 76, "rounds_underwater": 48, "abstain_rounds": 0, "backtest_score": 0.08494147889291231, "backtest_acc": 0.5483469605403484, "backtest_n": 4815}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus (coherence) as a multiplier for signal reliability.
    """
    df = df.copy()

    # 1. OBI Coherence: How much do the exchanges agree on passive intent?
    # OBI is [-1, 1], so max spread is 2.0.
    df['obi_agreement'] = 1.0 - (df['obi_spread'] / 2.0)

    # 2. Delta Coherence: How much do the exchanges agree on aggressive flow?
    # We measure the dispersion of delta relative to the total absolute flow.
    # High agreement = all exchanges seeing similar aggressive direction.
    delta_diff = (abs(df['binance_delta'] - df['coinbase_delta']) + 
                  abs(df['coinbase_delta'] - df['kraken_delta']))
    delta_sum = abs(df['binance_delta']) + abs(df['coinbase_delta']) + abs(df['kraken_delta'])
    df['delta_agreement'] = (delta_diff / (delta_sum + 1e-6))
    # We want 1.0 to be high agreement, so we invert the dispersion
    df['delta_agreement'] = 1.0 - df['delta_agreement']
    
    # Clip to ensure valid ranges [0, 1]
    df['obi_agreement'] = df['obi_agreement'].clip(0, 1)
    df['delta_agreement'] = df['delta_agreement'].clip(0, 1)

    # 3. Consensus Index: The combined signal of multi-venue agreement
    df['consensus_index'] = (df['obi_agreement'] + df['delta_agreement']) / 2.0

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1500,
        'max_depth': 3,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Venue Coherence Scaling (CVCS)".
    
    Logic:
    The model provides the direction (P(Up) vs P(Down)). 
    The consensus_index provides the reliability.
    - High EV + High Consensus = High conviction trend trade.
    - High EV + Low Consensus = Low conviction "noisy" trade (small size).
    - Low EV = Abstain.
    """
    # Exit if too little time remains for the move to resolve
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Get the consensus from engineered features
    consensus = features_row.get('consensus_index', 0.5)
    
    side = None
    bet_fraction = 0.0

    # --- REGIME: HIGH CONFIDENCE ---
    # If the model shows strong edge, we scale the bet by the consensus.
    # We use a tiered approach: 
    # 1. If EV is decent (>0.025), bet is proportional to consensus.
    # 2. If EV is massive (>0.05), we ignore the noise and bet full size.

    if ev_up > ev_down:
        if ev_up > 0.05:
            side = 'Up'
            bet_fraction = 1.0
        elif ev_up > 0.025:
            side = 'Up'
            # Scale bet: if consensus is 0.8, bet 80%. If 0.2, bet 20%.
            bet_fraction = consensus
        else:
            return None, 0.0
    else:
        if ev_down > 0.05:
            side = 'Down'
            bet_fraction = 1.0
        elif ev_down > 0.025:
            side = 'Down'
            bet_fraction = consensus
        else:
            return None, 0.0

    # Final safety: ensure we don't bet on low-consensus noise unless EV is huge
    if consensus < 0.2 and bet_fraction > 0.3:
        # If consensus is extremely low, we only allow a tiny "speculative" bet
        bet_fraction = 0.1

    return side, min(1.0, max(0.0, bet_fraction))