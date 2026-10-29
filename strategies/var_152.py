# Var 152 | BT: 1.022 | Live PnL: $-3.19 | Trades: 57
# Desc: Implements "Multi-Venue Flow-Efficiency Gating (MVFEG)," which filters for high-conviction trades by requiring both cross-exchange delta consensus (at least 2 of 3 exchanges agreeing on direction) and
# Stats: {"live_returns": [-0.3479698896408081, 0.006458573043346405, 0.19925826787948608, 0.19406256079673767, -0.8486441373825073, 0.002977795898914337, -0.3921230733394623, 1.519559383392334, -0.9038230180740356, 0.004301972687244415, -0.4775420129299164, 0.02188664674758911, 0.007650852203369141, -0.09674014151096344, 0.002073824405670166, 0.014748439192771912, 0.32369136810302734, 0.05674973130226135, -0.21024495363235474, 0.007538758218288422, -0.16481786966323853, -1.0, -0.09812605381011963, 0.01227916032075882, -0.1902095526456833, 0.02780473232269287, -0.053428903222084045, 0.07598602771759033, 0.003797486424446106, -0.0878405049443245, -0.605025053024292, 0.2644539177417755, 0.004096638411283493, 0.003677383065223694, 0.0014352999999999935, 0.0317104309797287, -0.05, -0.19869959354400635, -0.33526983857154846, -0.05, 0.01446966826915741, -0.12201735377311707, 0.6485327482223511, 0.0014352999999999935, 0.09935939311981201, 0.006428763270378113, -0.05, -0.20258963108062744, 1.1556274890899658, 0.004839695990085602, -0.8235349059104919, -0.20344296097755432, -0.428329735994339, 0.012677431106567383, 0.013091452419757843, 0.0045912999999999995, 0.004036600000000001], "live_pnl": -3.1891298294067383, "live_wins": 33, "live_total": 57, "age": 911, "rounds_underwater": 49, "abstain_rounds": 57, "backtest_score": 1.0217053189998238, "backtest_acc": 0.5747672430172905, "backtest_n": 486}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange delta consensus and price-flow efficiency.
    """
    df = df.copy()

    # 1. Delta Consensus: How many exchanges agree with the aggregate mean_delta sign?
    # We use np.sign to handle the directionality.
    mean_sign = np.sign(df['mean_delta'])
    agreement_count = (
        (np.sign(df['binance_delta']) == mean_sign).astype(int) +
        (np.sign(df['coinbase_delta']) == mean_sign).astype(int) +
        (np.sign(df['kraken_delta']) == mean_sign).astype(int)
    )
    df['delta_agreement'] = agreement_count

    # 2. Flow-Price Efficiency: Does the aggressive flow (delta) align with 
    # the recent price momentum? (1 if aligned, -1 if opposing, 0 if neutral)
    df['flow_efficiency'] = np.sign(df['mean_delta']) * np.sign(df['mean_momentum'])

    # 3. Volatility-Adjusted Spread: A measure of market noise.
    df['noise_index'] = df['price_spread'] + df['mean_volatility']

    # 4. Delta Magnitude: The strength of the aggressive flow.
    df['delta_magnitude'] = df['mean_delta'].abs()

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
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Multi-Venue Flow-Efficiency Gating (MVFEG)".

    Logic:
    1. Calculate Conviction:
       - Conviction = (Agreement Ratio) * (Efficiency Factor)
       - Agreement Ratio: 1.0 if all 3 exchanges agree, 0.66 if 2 agree, 0.33 if 1 agrees.
       - Efficiency Factor: 1.0 if Delta and Momentum align, 0.0 otherwise.
    2. Gating:
       - Only trade if Conviction >= 0.66 (requires consensus AND efficiency).
    3. Execution:
       - Scale bet size by the magnitude of the edge, the conviction score, 
         and inversely by the market noise (spread/volatility).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    agreement_ratio = features_row.get('delta_agreement', 0.0) / 3.0
    efficiency = 1.0 if features_row.get('flow_efficiency', 0.0) > 0 else 0.0
    noise = features_row.get('noise_index', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    price_spread = features_row.get('price_spread', 0.0)

    conviction = agreement_ratio * efficiency
    
    side = None
    best_ev = 0.0

    # 1. High-Conviction Gating
    # We only enter when at least 2 exchanges agree AND price/flow are efficient.
    if conviction >= 0.66:
        if mean_delta > 0 and ev_up > 0.02:
            side = 'Up'
            best_ev = ev_up
        elif mean_delta < 0 and ev_down > 0.02:
            side = 'Down'
            best_ev = ev_down

    # 2. Execution and Dynamic Sizing
    if side:
        # Scale bet fraction by:
        # a) Edge magnitude (target 4% EV for full bet)
        # b) Conviction (consensus + efficiency)
        # c) Noise penalty (higher spreads/volatility = smaller bets)
        
        # Ensure we don't divide by zero in noise penalty
        noise_penalty = 1.0 / (1.0 + noise)
        
        # Time decay: reduce size as the window closes to avoid last-second whipsaws
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = (best_ev / 0.04) * conviction * noise_penalty * time_factor
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0