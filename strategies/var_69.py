# Var 69 | BT: 0.322 | Live PnL: $-11.32 | Trades: 61
# Desc: Implements "Consensus-Compression Convergence (CCC)," which filters for high-conviction trades by requiring multi-exchange delta agreement combined with narrowing price spreads (indicating liquidity a
# Stats: {"live_returns": [-1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.04824608564376831, 0.05401700735092163, -0.6948103904724121, 5.282196, -1.0, 0.576209545135498, 2.950496, -1.0, 0.06178271770477295, 3.505625, -1.0, -0.5708155632019043, 0.08630067110061646, -0.7761383056640625, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -0.8506393432617188, -1.0, -1.0, -1.0, -1.0, -0.5547845363616943, -1.0, -1.0, -0.5255672931671143, 2.39541, 0.02651691436767578, 0.5715265274047852, -0.5255672931671143, -1.0, 0.6337271928787231, 1.1621617078781128, -1.0, -1.0, -0.5255672931671143, -1.0, -1.0, -1.0, 1.9791409999999998, -0.4979568421840668, -0.604686975479126, -1.0, -1.0, -1.0], "live_pnl": -11.315335273742676, "live_wins": 17, "live_total": 61, "age": 79, "rounds_underwater": 27, "abstain_rounds": 0, "backtest_score": 0.3216380610096745, "backtest_acc": 0.5703270406806701, "backtest_n": 2757}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus and liquidity compression.
    """
    df = df.copy()

    # 1. Consensus Delta: Do all exchanges agree on the direction of aggressive flow?
    # Returns values in {-3, -2, -1, 0, 1, 2, 3}
    df['consensus_delta'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    )

    # 2. Spread Compression: Is the cross-exchange price dispersion narrowing?
    # A negative diff in spread means the spread is shrinking (compression).
    # We use -diff so that positive values = compression.
    df['spread_compression'] = -df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 3. Delta Velocity: Is the aggressive flow accelerating?
    df['delta_velocity'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)

    # 4. OBI Velocity: Is the order book imbalance shifting rapidly?
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

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
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Consensus-Compression Convergence (CCC)".

    Logic:
    - High Conviction: All 3 exchanges agree on delta sign AND price spreads are narrowing.
      (Indicates a clean, high-liquidity move). Requires low EV threshold.
    - Low Conviction: Exchanges disagree OR spreads are widening.
      (Indicates fragmentation or volatility). Requires high EV threshold.
    - Neutral: Default state.
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
    consensus = features_row.get('consensus_delta', 0.0)
    comp = features_row.get('spread_compression', 0.0)
    
    # 1. Define Market States
    # High Conviction: Absolute consensus (3 or -3) and narrowing spreads
    is_high_conviction = (abs(consensus) == 3) and (comp > 0)
    
    # Low Conviction: Disagreement (abs < 3) or widening spreads (comp < 0)
    is_low_conviction = (abs(consensus) < 3) or (comp < 0)

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_high_conviction:
        req_ev = 0.020
        multiplier = 1.0
    elif is_low_conviction:
        req_ev = 0.050
        multiplier = 0.4
    else:
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating
    # Increase required edge as the window closes to avoid late-stage noise
    if seconds_remaining < 60:
        req_ev *= 1.5
        multiplier *= 0.8

    # 4. Final Decision and Dynamic Sizing
    if best_ev > req_ev:
        # Scale bet size based on how much the edge exceeds the target 5% threshold
        bet_fraction = multiplier * (best_ev / 0.05)
        return side, min(1.0, bet_fraction)

    return None, 0.0