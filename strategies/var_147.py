# Var 147 | BT: -0.210 | Live PnL: $-21.25 | Trades: 60
# Desc: Implements "Liquidity-Gap Vacuum (LGV)," which identifies periods of high cross-exchange price dispersion and order book instability to capture rapid price jumps when aggressive flow (Delta) enters a
# Stats: {"live_returns": [-1.0, 0.15118699999999996, -1.0, -1.0, -1.0, -1.0, 1.092555, 0.16389999999999993, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, -1.0, -1.0, 0.10316799999999993, 2.651597, -1.0, 0.04885600000000001, -1.0, -1.0, -1.0, 0.028705999999999898, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, 1.5191249999999998, 0.04885600000000001, 0.018941000000000097, -1.0, -1.0, 1.049452, -1.0, 1.9791409999999998, 1.233902, -1.0, 1.1846420000000002, -1.0, 1.1375739999999999, -1.0, -1.0, 1.092555, -1.0, -1.0, -1.0, 0.12663000000000002], "live_pnl": -21.253207999999997, "live_wins": 21, "live_total": 60, "age": 179, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": -0.21028346542677423, "backtest_acc": 0.5801941747572815, "backtest_n": 2698}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'Vacuum Effect'—where price dispersion and book thinness 
    suggest imminent liquidity-driven price jumps.
    """
    df = df.copy()

    # 1. Vacuum Index: Normalized measure of market fragmentation.
    # High value = Exchanges are far apart (price_spread) AND the book is thin (obi_spread)
    # relative to current volatility. This indicates a "liquidity vacuum."
    vol = df['mean_volatility'].clip(lower=1e-6)
    # Normalize spread by price and volatility to make it scale-invariant
    price_gap_norm = (df['price_spread'] / (df['binance_price'] * vol + 1e-9))
    obi_gap_norm = (df['obi_spread'] / vol)
    df['vacuum_index'] = price_gap_norm * obi_gap_norm

    # 2. Flow Coherence: How well the aggressive flow (Delta) aligns with intent (OBI).
    df['flow_coherence'] = df['mean_delta'] * df['mean_obi']

    # 3. Volatility-Adjusted Delta: Magnitude of flow relative to market noise.
    df['adj_delta'] = df['mean_delta'] / vol

    # 4. Momentum-Volatility Ratio: Is the momentum sustainable or just noise?
    df['momentum_efficiency'] = df['mean_momentum'] / vol

    # Drop metadata and target
    # We drop seconds_remaining here because it's passed separately to strategy_filter
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
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
    Decide whether to trade using "Liquidity-Gap Vacuum (LGV)" logic.

    Logic:
    1. Identify Vacuum Regime: If 'vacuum_index' is high, the market is fragmented.
       In this regime, aggressive flow (Delta) is likely to cause a disproportionate 
       price jump to close the spread.
    2. Vacuum Execution: 
       - If in a vacuum, we lower the EV threshold (0.015) because the 'jump' 
         provides the alpha, and we follow the direction of the Delta.
       - We amplify the bet size based on the magnitude of the vacuum.
    3. Standard Execution:
       - In normal regimes, we use a strict EV threshold (0.04) to preserve Sharpe.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    vacuum_index = features_row.get('vacuum_index', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    
    side = None
    best_ev = 0.0

    # A vacuum is more impactful as the market window approaches its close
    time_urgency = 1.0 + (1.0 - (seconds_remaining / 300.0)) 
    effective_vacuum = vacuum_index * time_urgency

    # Threshold for entering the Vacuum Regime
    # (Adjusted based on normalized scale)
    VACUUM_THRESHOLD = 0.5 

    if effective_vacuum > VACUUM_THRESHOLD:
        # --- VACUUM REGIME ---
        # We trade the aggressive flow direction, even with lower model confidence,
        # because the liquidity gap implies a high-probability price jump.
        if mean_delta > 0 and ev_up > 0.015:
            side = 'Up'
            best_ev = ev_up
        elif mean_delta < 0 and ev_down > 0.015:
            side = 'Down'
            best_ev = ev_down
            
        if side:
            # Scale bet: Higher vacuum = higher conviction in the jump
            # Using log1p to prevent explosive bet sizes
            bet_multiplier = 1.0 + math.log1p(effective_vacuum)
            bet_fraction = (best_ev / 0.04) * bet_multiplier
            return side, min(1.0, max(0.1, bet_fraction))

    else:
        # --- NORMAL REGIME ---
        # Standard high-conviction thresholding
        if ev_up > 0.04:
            side = 'Up'
            best_ev = ev_up
        elif ev_down > 0.04:
            side = 'Down'
            best_ev = ev_down

        if side:
            bet_fraction = (best_ev / 0.04)
            return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0