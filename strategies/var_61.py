# Var 61 | BT: 0.074 | Live PnL: $-0.02 | Trades: 60
# Desc: Implements "Tri-Exchange Microstructure Congruence (TEMC)," which filters model predictions by requiring directional alignment across all three major exchanges (Binance, Coinbase, Kraken) in both aggr
# Stats: {"live_returns": [-0.1873170156478882, -0.1874559586842855, 1.1677115621461025, -0.18741674264272057, -0.1870031785964966, 0.5545880104836934, -0.18737594064076743, -0.16114267921447756, -0.18792639350891116, 0.3853101053951549, 0.04296690576254203, -0.18546873982747397, 0.23189446137286382, -0.18762470467885337, -0.18777202860514325, -0.1861722494761149, 0.02156692008366204, -0.18767382876078287, 0.324484266635954, -0.1613106654485067, -0.1874874887466431, -0.18792856947580974, 0.2978566347194061, 0.6960961692952315, 0.5246111742370644, -0.16079025046030682, -0.18786668809254967, -0.18774260203043622, 0.007264895635827384, -0.1878045721054077, 0.007264139330355329, 0.38664027525300504, 0.42693847368285087, -0.18441026178995767, 0.028369165654938383, 0.017250164883287428, 0.22238640435354934, 0.3248804538315888, -0.18788427511850994, -0.18782880179087322, -0.1880861380894979, 0.31828663908004057, -0.18790658950805664, -0.18761357561747236, -0.18784864711761476, -0.18784730116526285, -0.1878258571624756, -0.18739049943288166, -0.18694687430063883, -0.17908910210927328, 0.009171640767931616, -0.1874286114374797, -0.18769280338287353, 0.31250320464521886, -0.1342258342107137, 0.02155960241632876, -0.18786435317993166, 0.2320434559587905, 0.028269879882150017, -0.187647274017334], "live_pnl": -0.01690249056688492, "live_wins": 24, "live_total": 60, "age": 408, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.07388820018510842, "backtest_acc": 0.5666846652267818, "backtest_n": 574}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus of order flow and liquidity.
    """
    df = df.copy()

    # 1. Consensus Scores: Sum of signs across the three exchanges
    # Result is in {-3, -2, -1, 0, 1, 2, 3}
    # 3 = perfect bullish consensus, -3 = perfect bearish consensus
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

    # 2. Liquidity-to-Volatility Ratio
    # Measures how much "room" there is to move before volatility dominates the spread
    df['spread_vol_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 3. Flow-to-Liquidity Ratio
    # Is the aggressive delta actually being supported by the OBI?
    df['flow_obi_ratio'] = df['mean_delta'] / (df['mean_obi'] + 1e-6)

    # 4. Price-Strike Momentum
    # Velocity of price relative to the market start price
    df['price_velocity'] = df['price_vs_strike_pct']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for capturing consensus-driven signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Exchange Microstructure Congruence (TEMC)".
    
    A trade is only taken if the model's predicted direction is corroborated by 
    the consensus of the three exchanges.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine the model's preferred side
    if ev_up > ev_down:
        target_side = 'Up'
        target_sign = 1
        target_ev = ev_up
    elif ev_down > ev_up:
        target_side = 'Down'
        target_sign = -1
        target_ev = ev_down
    else:
        return None, 0.0

    # Minimum hurdle for any trade
    hurdle = 0.035
    if target_ev < hurdle:
        return None, 0.0

    # Extract consensus features
    delta_cons = features_row.get('delta_consensus', 0.0)
    obi_cons = features_row.get('obi_consensus', 0.0)

    # Define Congruence Levels
    # Perfect: All 3 exchanges agree on Delta AND all 3 agree on OBI, and both match target
    is_perfect = (
        abs(delta_cons) == 3 and 
        abs(obi_cons) == 3 and 
        np.sign(delta_cons) == target_sign and 
        np.sign(obi_cons) == target_sign
    )
    
    # Partial: At least one component (Delta or OBI) shows majority agreement (>= 2/3)
    is_partial = (
        (abs(delta_cons) >= 2 and np.sign(delta_cons) == target_sign) or
        (abs(obi_cons) >= 2 and np.sign(obi_cons) == target_sign)
    )

    # Decision Logic
    if is_perfect:
        confidence = 1.0
    elif is_partial:
        confidence = 0.4
    else:
        # If there is no consensus or the consensus is against the model, abstain
        return None, 0.0

    # Time-decay factor: Scale exposure as the window closes to manage late-window noise
    # We want to be more cautious in the final 60 seconds
    time_factor = min(1.0, seconds_remaining / 300.0)
    
    # Final bet sizing
    bet_fraction = confidence * time_factor

    # Safety threshold
    if bet_fraction < 0.1:
        return None, 0.0

    return target_side, min(1.0, bet_fraction)