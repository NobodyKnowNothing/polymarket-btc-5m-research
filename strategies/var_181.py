# Var 181 | BT: 1.044 | Live PnL: $-0.01 | Trades: 83
# Desc: Implements "Cross-Venue Coherence & Liquidity-Gated Execution (CVCLGE)," which quantifies the consensus level across all three exchanges and scales conviction based on the ratio of market spread to vo
# Stats: {"live_returns": [0.47139385000000006, -0.05, -0.10330633074045181, -0.6613138318061829, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, 0.0792714, 0.30413344502449036, 0.08647035000000002, -0.20927736163139343, -0.05, -0.05, 0.05687869999999999, -0.05, -0.05, 0.05923210000000001, 0.1259379, 0.05247260000000001, 0.0792714, -0.05, 0.1475248, 0.06698185000000001, -0.47744715213775635, 0.020078659057617188, -0.13001343607902527, -0.7795174717903137, 0.10866970000000002, -0.21127338707447052, 2.1816108226776123, -0.17048285901546478, -0.05, -0.05, 2.6273598670959473, 0.0029626999999999987, 0.020205974578857422, 0.006331500000000004, -0.05, 0.0616951, -0.05795643851161003, -0.05, -0.5863959193229675, 0.007930479943752289, 0.29409468173980713, -0.8642090559005737, 0.005738350000000003, 0.008194999999999994, -0.05, 0.004036600000000001, -0.05, -0.05, -0.05, -0.05, -0.05, -0.38197433948516846, -0.0664789006114006, 0.01988927274942398, 0.08277080000000002, 0.1298547089099884, 0.1259379, -0.05, -0.6385360360145569, 0.07281125, 0.0045912999999999995, -0.05, 0.008845899999999997, -0.05, -0.20135332643985748, -0.05219985172152519, 0.859374463558197, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.10050255805253983, -0.573896050453186, -0.3259652853012085], "live_pnl": -0.009545683860778809, "live_wins": 32, "live_total": 83, "age": 288, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 1.0435005600129685, "backtest_acc": 0.5703958069763239, "backtest_n": 1921}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-venue consensus and the relationship between spread and volatility.
    """
    X = df.copy()

    # 1. Cross-Venue OBI Agreement
    # Measures how many exchanges agree on the direction of order book imbalance.
    # 1.0 = all agree, 0.33 = no agreement/mixed, 0.0 = complete opposition.
    s_obi_b = np.sign(X['binance_obi'])
    s_obi_c = np.sign(X['coinbase_obi'])
    s_obi_k = np.sign(X['kraken_obi'])
    
    pos_obi = ((s_obi_b > 0).astype(int) + (s_obi_c > 0).astype(int) + (s_obi_k > 0).astype(int))
    neg_obi = ((s_obi_b < 0).astype(int) + (s_obi_c < 0).astype(int) + (s_obi_k < 0).astype(int))
    X['obi_agreement'] = np.maximum(pos_obi, neg_obi) / 3.0

    # 2. Cross-Venue Delta Agreement
    # Measures how many exchanges agree on the direction of aggressive flow.
    s_del_b = np.sign(X['binance_delta'])
    s_del_c = np.sign(X['coinbase_delta'])
    s_del_k = np.sign(X['kraken_delta'])
    
    pos_del = ((s_del_b > 0).astype(int) + (s_del_c > 0).astype(int) + (s_del_k > 0).astype(int))
    neg_del = ((s_del_b < 0).astype(int) + (s_del_c < 0).astype(int) + (s_del_k < 0).astype(int))
    X['delta_agreement'] = np.maximum(pos_del, neg_del) / 3.0

    # 3. Microstructure Coherence
    # High coherence means the intent (OBI) matches the action (Delta).
    X['coherence_index'] = (np.sign(X['mean_obi']) == np.sign(X['mean_delta'])).astype(float)

    # 4. Volatility-Adjusted Spread (Noise Index)
    # Higher values indicate high spread relative to volatility (noisy/unreliable regime).
    X['vol_spread_ratio'] = (X['price_spread'] + X['obi_spread']) / (X['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

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
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using "Cross-Venue Coherence & Liquidity-Gating".

    Logic:
    1. Consensus Check: Require a minimum level of agreement across exchanges 
       (obi_agreement and delta_agreement) and alignment between intent and action (coherence).
    2. Edge Validation: Calculate EV for both sides.
    3. Risk Scaling: 
       - Scale by 'confidence' (agreement * coherence).
       - Scale by 'liquidity quality' (inverse of vol_spread_ratio).
       - Scale by 'time decay' (reduce size as the window closes).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    obi_agreement = features_row.get('obi_agreement', 0.0)
    delta_agreement = features_row.get('delta_agreement', 0.0)
    coherence = features_row.get('coherence_index', 0.0)
    vol_spread_ratio = features_row.get('vol_spread_ratio', 0.0)

    # 1. Calculate Combined Confidence Score
    # We need both venue agreement and intent-action coherence.
    agreement_score = (obi_agreement + delta_agreement) / 2.0
    confidence = agreement_score * coherence

    # 2. Selection Logic
    side = None
    best_ev = 0.0
    
    # Minimum confidence threshold to avoid low-conviction noise
    if confidence > 0.5:
        if ev_up > 0.03 and ev_up > ev_down:
            side = 'Up'
            best_ev = ev_up
        elif ev_down > 0.03 and ev_down > ev_up:
            side = 'Down'
            best_ev = ev_down

    # 3. Dynamic Bet Sizing
    if side:
        # Target 4% EV for a full bet
        base_bet = best_ev / 0.04
        
        # Scale by confidence (0.5 to 1.0)
        # Scale by liquidity quality (1 / (1 + noise))
        # Scale by time (avoiding late-window volatility spikes)
        time_factor = 1.0 if seconds_remaining > 45 else (seconds_remaining / 45.0)
        liquidity_factor = 1.0 / (1.0 + vol_spread_ratio)
        
        bet_fraction = base_bet * confidence * liquidity_factor * time_factor
        
        # Clamp bet between 5% and 100%
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0