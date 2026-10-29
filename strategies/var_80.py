# Var 80 | BT: 0.006 | Live PnL: $-0.36 | Trades: 197
# Desc: Implements "Lead-Exchange Consensus Gating (LECG)," which amplifies expected value when Binance (the liquidity leader) shows price leadership and all three exchanges reach a consensus on delta directi
# Stats: {"live_returns": [-1.0, 1.8871363401412964, 2.173394, 2.073133, 1.233902, -1.0, 1.5191249999999998, 3.303756, -1.0, 0.15118699999999996, -1.0, 1.0099737644195557, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, 2.950496, 1.4562249999999999, -1.0, -1.0, 1.5854279999999998, 1.8378456830978394, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, 0.09182599999999996, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -0.8955434560775757, 1.233902, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, 0.11476700000000006, 2.651597, -1.0, -1.0, -1.0, -0.944477379322052, -1.0, -0.8986963629722595, 4.550646, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 3.727685, 2.795066, 1.5368645191192627, 2.795066, 3.727685, 3.1194439999999997, 1.3396370000000002, -1.0, 1.9791409999999998, -0.7356798052787781, -1.0, -1.0, 3.303756, -1.0, -1.0, -0.7356798052787781, -1.0, 2.795066, -1.0, 2.2146122455596924, -1.0, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -1.0, 0.038675000000000015, -1.0, 0.09182599999999996, -1.0, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.912255048751831, 1.5808119773864746, -1.0, -0.8898616433143616, -1.0, 1.7294070000000001, -1.0, -1.0, 1.3858294486999512, -1.0, -1.0, -1.0, 2.280573, 1.9791409999999998, 1.890848, -1.0, -1.0, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, 1.8077519999999998, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -0.8499565720558167, -1.0, -1.0, 1.233902, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, 2.950496, -1.0, -1.0, -1.0, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -0.7308211922645569, -1.0, 2.073133, 2.651597, -1.0, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, -1.0, -0.7753547430038452, -0.9969955682754517, 1.5191249999999998, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -1.0, 1.1375739999999999, -1.0, -1.0, -1.0], "live_pnl": -0.3595921993255615, "live_wins": 66, "live_total": 197, "age": 592, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.005944234982225571, "backtest_acc": 0.563953488372093, "backtest_n": 1747}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, emphasizing 
    Binance price leadership and cross-exchange delta unanimity.
    """
    df = df.copy()

    # 1. Binance Price Lead: Is Binance price leading the other two exchanges?
    # Positive = Binance is higher (leading bullishly), Negative = Binance is lower.
    df['binance_price_lead'] = df['binance_price'] - (df['coinbase_price'] + df['kraken_price']) / 2

    # 2. Delta Consensus: Unanimity of aggressive flow across all three exchanges.
    # Range: [-3, 3]. 3 = all buying, -3 = all selling, 0 = total disagreement/neutrality.
    df['delta_consensus'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    )

    # 3. Price Velocity: Rate of price change relative to time remaining.
    # Captures the "speed" of the move towards the strike.
    df['price_velocity'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1e-6)

    # 4. OBI Unanimity: Do the order books agree on direction?
    df['obi_consensus'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    )

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
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Exchange Consensus Gating (LECG)".

    Logic:
    - Calculate raw EV for both sides.
    - Consensus Factor: Scales EV by the unanimity of delta signs (0.0 to 1.0).
    - Lead Factor: Amplifies EV (1.2x) if the trade direction aligns with 
      Binance's price leadership, or penalizes it (0.7x) if it contradicts.
    - Effective EV: The product of raw EV, consensus, and lead factors.
    - Bet Sizing: Scales bet fraction by the strength of the consensus.
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
    delta_consensus = features_row.get('delta_consensus', 0.0)
    binance_lead = features_row.get('binance_price_lead', 0.0)

    # 1. Consensus Factor (0.0 to 1.0)
    # High consensus (3 or -3) = 1.0. No consensus (0) = 0.0.
    consensus_factor = abs(delta_consensus) / 3.0

    # 2. Lead Factor (0.7 to 1.2)
    # If we are going 'Up', we want Binance price to be leading (positive).
    # If we are going 'Down', we want Binance price to be lagging (negative).
    lead_factor = 1.0
    if side == 'Up':
        lead_factor = 1.2 if binance_lead > 0 else 0.7
    else: # side == 'Down'
        lead_factor = 1.2 if binance_lead < 0 else 0.7

    # 3. Calculate Effective EV
    effective_ev = best_ev * consensus_factor * lead_factor

    # 4. Dynamic Thresholding
    # We require a higher effective edge to compensate for the multi-factor scaling.
    base_threshold = 0.035
    if seconds_remaining < 60:
        base_threshold = 0.05

    if effective_ev > base_threshold:
        # Scale bet fraction based on effective EV and consensus strength.
        # We target a 4% effective edge for a full bet, but scale down if consensus is weak.
        bet_fraction = (effective_ev / 0.04) * (consensus_factor + 0.5)
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0