# Var 175 | BT: -0.435 | Live PnL: $-11.61 | Trades: 60
# Desc: Implements "Cross-Venue Consensus & Volatility-Adjusted Hurdle (CVC-VAH)," which scales the required Expected Value (EV) threshold based on the level of agreement between Binance, Coinbase, and Kraken
# Stats: {"live_returns": [-0.9381093382835388, 0.02400106191635132, -1.0, 4.245841, -0.8191863894462585, 2.518758, -1.0, -1.0, -1.0, 2.518758, 1.1846420000000002, -1.0, -1.0, -0.8191863894462585, -1.0, -0.39681968092918396, -0.7053168416023254, -0.9889697432518005, 1.4562249999999999, -1.0, 1.174009084701538, 2.280573, -1.0, -1.0, 0.47635531425476074, -1.0, -1.0, 0.03792858123779297, -0.6539689302444458, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, 6.165539264678955, 1.3643214702606201, -0.43680083751678467, -1.0, -1.0, -1.0, -0.6555600166320801, -0.8036637306213379, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.33167678117752075, -1.0, -1.0, -0.7389957904815674, -1.0, 1.6554160000000002, -0.7389957904815674, -0.6719939708709717, 0.06817787885665894, 0.12663000000000002], "live_pnl": -11.607001304626465, "live_wins": 16, "live_total": 60, "age": 86, "rounds_underwater": 19, "abstain_rounds": 0, "backtest_score": -0.43461693082098973, "backtest_acc": 0.5902151922015818, "backtest_n": 3203}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus and the alignment of intent (OBI) and action (Delta).
    """
    df = df.copy()

    # 1. Delta Consensus: How many exchanges agree with the aggregate delta sign?
    # This measures the strength of the directional signal across venues.
    def get_consensus(row):
        target_sign = np.sign(row['mean_delta'])
        if target_sign == 0:
            return 0
        matches = 0
        for col in ['binance_delta', 'coinbase_delta', 'kraken_delta']:
            if np.sign(row[col]) == target_sign:
                matches += 1
        return matches

    df['delta_consensus'] = df.apply(get_consensus, axis=1)

    # 2. OBI-Delta Alignment: Do the limit orders (intent) support the aggressive flow (action)?
    df['obi_delta_alignment'] = df['mean_obi'] * df['mean_delta']

    # 3. Volatility-Normalized Delta: Scaling the flow magnitude by the current volatility regime.
    # High volatility requires larger delta to be considered significant.
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price-Flow Divergence: Is the price moving significantly relative to the delta?
    # Low values suggest high absorption (heavy walls); high values suggest a vacuum.
    df['price_flow_efficiency'] = df['mean_delta'] / (df['price_vs_strike_pct'] + 1e-6)

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
    Decide whether to trade using "Cross-Venue Consensus & Volatility-Adjusted Hurdle (CVC-VAH)".

    Logic:
    1. Calculate Consensus Ratio: (Number of agreeing exchanges) / 3.
    2. Dynamic Hurdle: The required EV (expected value) increases as consensus decreases.
       - If all 3 agree, hurdle is low (~2.5%).
       - If only 1 agrees, hurdle is high (~6%).
    3. Volatility Scaling: Increase the hurdle during high-volatility regimes to avoid noise.
    4. Sizing: Scale bet size linearly with the consensus ratio and the magnitude of the edge.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine best available edge
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    consensus_count = features_row.get('delta_consensus', 0.0)
    consensus_ratio = consensus_count / 3.0
    mean_vol = features_row.get('mean_volatility', 0.0)

    # 1. Calculate Dynamic Hurdle
    # Base hurdle is 3%. We divide by (consensus + offset) to make it harder to trade when consensus is low.
    # We also multiply by a volatility factor.
    vol_multiplier = 1.0 + (mean_vol * 5.0) # Assuming mean_vol is a small float
    base_hurdle = 0.03
    adjusted_hurdle = (base_hurdle * vol_multiplier) / (consensus_ratio + 0.5)

    # 2. Execution Logic
    if best_ev > adjusted_hurdle:
        # Scale bet fraction:
        # - Proportional to the edge (target 4% EV for full bet)
        # - Proportional to consensus (don't bet big if exchanges disagree)
        # - Decay as time runs out to avoid late-window volatility spikes
        
        time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        # We want consensus_ratio=1.0 to allow full sizing, consensus_ratio=0.33 to reduce it.
        bet_fraction = (best_ev / 0.04) * consensus_ratio * time_factor
        
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0