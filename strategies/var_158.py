# Var 158 | BT: 0.243 | Live PnL: $-6.05 | Trades: 60
# Desc: Implements "Cross-Venue Consensus Gating (CVCG)," which requires high agreement across all three exchanges for both order book intent (OBI) and aggressive flow (Delta) before entering a trade, effecti
# Stats: {"live_returns": [1.760586142539978, -1.0, 0.01466698944568634, -0.3380066156387329, 0.15692728757858276, 2.3384764194488525, -0.9876047968864441, -0.5917564630508423, -1.0, -1.0, -0.3710303008556366, 2.07047700881958, -1.0, 0.491202175617218, 0.5506721138954163, 1.3821734189987183, -0.412814736366272, 0.4251502752304077, 0.4959834814071655, -0.16596084833145142, -1.0, -0.5107806324958801, 0.0438380241394043, -1.0, -1.0, 0.3189765214920044, -1.0, -0.4376829266548157, -0.28997787833213806, -1.0, -0.9068095088005066, 0.2309383898973465, -1.0, 0.740162193775177, -0.6796445250511169, -0.4327750504016876, -0.6185624003410339, -0.4488776922225952, -0.4971279501914978, -0.2586255669593811, -0.6750074028968811, -0.930296778678894, -0.6535707712173462, -1.0, 1.088390827178955, -0.29065629839897156, 3.1194439999999997, -1.0, -0.6102347373962402, -0.7246378660202026, -1.0, 0.3225111663341522, -1.0, -0.8986713290214539, 0.7063531875610352, 4.245841, -0.14474867284297943, 1.6198923587799072, -0.9147478342056274, -0.3809089958667755], "live_pnl": -6.048855304718018, "live_wins": 20, "live_total": 60, "age": 96, "rounds_underwater": 36, "abstain_rounds": 0, "backtest_score": 0.2428503295435705, "backtest_acc": 0.5251865671641791, "backtest_n": 3715}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange agreement (consensus) and volatility-normalized flow.
    """
    df = df.copy()

    # 1. Delta Agreement: Sum of signs of delta across exchanges [-3, 3]
    # 3 = all exchanges buying, -3 = all selling, 0 = mixed/neutral
    df['delta_agreement'] = (np.sign(df['binance_delta']) + 
                             np.sign(df['coinbase_delta']) + 
                             np.sign(df['kraken_delta']))

    # 2. OBI Agreement: Sum of signs of OBI across exchanges [-3, 3]
    df['obi_agreement'] = (np.sign(df['binance_obi']) + 
                           np.sign(df['coinbase_obi']) + 
                           np.sign(df['kraken_obi']))

    # 3. Volatility-Normalized Delta: Magnitude of flow relative to current volatility
    df['vol_delta_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Spread/Volatility Ratio: Measures if dispersion is driven by volatility or noise
    df['price_spread_vol_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 5. Rolling Delta Stability: Standard deviation of delta within the market window
    # Grouping by market_slug to prevent leakage between different 5-min windows
    df['rolling_delta_std'] = df.groupby('market_slug')['mean_delta'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    ).fillna(0)

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
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Cross-Venue Consensus Gating (CVCG)".

    Logic:
    1. Consensus Score: Calculate how much the exchanges agree on Delta and OBI.
       Score ranges from 0.0 (total disagreement) to 1.0 (perfect consensus).
    2. Dynamic Hurdle: Require a higher Expected Value (EV) when consensus is low.
       If consensus is 1.0, EV threshold is ~2.5%. If consensus is 0.0, threshold is ~7.5%.
    3. Time/Risk Guard: 
       - Abstain if the market is too close to expiry (< 30s) to avoid last-second whipsaws.
       - Scale bet size by the consensus score and the remaining time in the window.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Safety: Avoid the "last second" volatility spike
    if seconds_remaining < 30:
        return None, 0.0

    # Extract engineered consensus features
    # agreement is in range [-3, 3], so we normalize to [0, 1]
    delta_aggr = features_row.get('delta_agreement', 0.0)
    obi_aggr = features_row.get('obi_agreement', 0.0)
    
    # consensus_score: 1.0 means all 3 exchanges agree on both Delta and OBI
    # Max possible sum of abs(aggr) is 3 + 3 = 6
    consensus_score = (abs(delta_aggr) + abs(obi_aggr)) / 6.0

    # Identify best side
    if ev_up > ev_down:
        side, best_ev = 'Up', ev_up
    else:
        side, best_ev = 'Down', ev_down

    # 1. Dynamic EV Thresholding
    # High consensus = lower bar (0.025); Low consensus = high bar (0.075)
    min_ev_threshold = 0.025 + (1.0 - consensus_score) * 0.05

    if best_ev > min_ev_threshold:
        # 2. Dynamic Bet Sizing
        # Scale by:
        # a) Magnitude of edge (relative to a 5% target)
        # b) Consensus strength (don't bet big on disagreement)
        # c) Time decay (reduce exposure as the window closes)
        
        time_factor = seconds_remaining / 300.0  # Linear decay over 5 mins
        edge_factor = best_ev / 0.05
        
        bet_fraction = edge_factor * consensus_score * time_factor
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0