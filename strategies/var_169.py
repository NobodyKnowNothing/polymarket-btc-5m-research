# Var 169 | BT: 1.233 | Live PnL: $-16.13 | Trades: 60
# Desc: Implements "Cross-Exchange Delta-Imbalance Divergence (CDID)," which distinguishes between "Consensus" regimes (where all exchanges move together) and "Fragmentation" regimes (where exchanges disagree
# Stats: {"live_returns": [-1.0, 1.3964720000000002, -1.0, -1.0, -0.9112873077392578, -1.0, -0.7255772352218628, -1.0, 1.28551, -0.912334680557251, -1.0, -1.0, -1.0, -0.743375301361084, -1.0, -1.0, -1.0, 1.890848, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -0.7255772352218628, 1.1352145671844482, -0.8221257925033569, -1.0, -1.0, 1.4562249999999999, -0.743375301361084, -0.721037745475769, 4.245841, -1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, 0.871529221534729, -1.0, -1.0, -1.0, -0.9200201034545898, -1.0, -1.0, 1.4562249999999999, 1.1257314682006836, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0], "live_pnl": -16.128515243530273, "live_wins": 14, "live_total": 60, "age": 93, "rounds_underwater": 58, "abstain_rounds": 0, "backtest_score": 1.2331595431318048, "backtest_acc": 0.5332096474953618, "backtest_n": 3728}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange fragmentation and liquidity-provider leadership.
    """
    df = df.copy()

    # 1. Delta Fragmentation: High value means exchanges are fighting/diverging
    # We use the standard deviation of delta across the three exchanges
    delta_cols = ['binance_delta', 'coinbase_delta', 'kraken_delta']
    df['delta_std'] = df[delta_cols].std(axis=1)
    df['delta_fragmentation'] = df['delta_std'] / (abs(df['mean_delta']) + 1e-6)

    # 2. OBI Consensus: Are all books leaning the same way?
    # High value = high agreement on intent
    df['obi_consensus'] = (abs(df['binance_obi']) + abs(df['coinbase_obi']) + abs(df['kraken_obi'])) / 3.0

    # 3. Price Dispersion Velocity: Is the gap between exchanges widening?
    # Widening spreads often precede high-volatility breakouts
    df['price_spread_velocity'] = df['price_spread'].diff().fillna(0)

    # 4. Leader-Follower Divergence: Does the lead exchange (Binance) 
    # disagree with the aggregate flow?
    df['binance_flow_divergence'] = df['binance_delta'] - df['mean_delta']

    # 5. Volatility-Adjusted Fragmentation
    df['vol_adj_fragmentation'] = df['delta_fragmentation'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to fragmented noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.015,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Exchange Delta-Imbalance Divergence (CDID)".

    Logic:
    1. Identify Market Regime:
       - CONSENSUS: Low delta fragmentation. All exchanges are in sync.
         Trade the model's predicted direction if OBI consensus is high.
       - FRAGMENTATION: High delta fragmentation. Exchanges are diverging.
         This is a lead-lag opportunity. We only trade if the 'Leader' (Binance) 
         agrees with the model's predicted side.
    2. Execution:
       - In Consensus: Standard EV-based entry.
       - In Fragmentation: Require the Leader's delta to align with the model's side.
       - Scale bet size by the magnitude of the edge and time remaining.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Determine model's preferred side
    if ev_up > ev_down:
        model_side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        model_side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    frag = features_row.get('delta_fragmentation', 0.0)
    obi_con = features_row.get('obi_consensus', 0.0)
    binance_delta = features_row.get('binance_delta', 0.0)
    
    # 1. Regime Detection
    # Thresholds: frag < 1.0 is consensus; frag > 2.0 is high fragmentation
    is_consensus = frag < 1.2
    is_fragmented = frag > 1.8

    # 2. Filtering Logic
    can_trade = False
    
    if is_consensus:
        # In consensus, we need decent OBI agreement to confirm the trend
        if obi_con > 0.2 and best_ev > 0.025:
            can_trade = True
    elif is_fragmented:
        # In fragmentation, we only bet if the leader (Binance) confirms the model
        leader_agrees = False
        if model_side == 'Up' and binance_delta > 0:
            leader_agrees = True
        elif model_side == 'Down' and binance_delta < 0:
            leader_agrees = True
            
        if leader_agrees and best_ev > 0.035:
            can_trade = True
    else:
        # Neutral zone: standard EV check
        if best_ev > 0.04:
            can_trade = True

    if can_trade:
        # 3. Dynamic Bet Sizing
        # Base size on EV, but penalize if we are in a high-fragmentation regime
        # because fragmentation increases the risk of a "fake-out"
        regime_multiplier = 1.0 if is_consensus else 0.7
        
        # Time safety: reduce size as we approach the close to avoid last-second whipsaws
        time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        # Target edge of 4% for full sizing
        bet_fraction = (best_ev / 0.04) * regime_multiplier * time_factor
        
        return model_side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0