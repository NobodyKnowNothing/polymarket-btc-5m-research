# Var 117 | BT: 0.133 | Live PnL: $-0.74 | Trades: 116
# Desc: Implements "Squeeze & Absorption Congruence (SAC) Gating," which amplifies bets that align with high-conviction microstructure signals (Funding Squeezes and Order Flow Absorption) while aggressively p
# Stats: {"live_returns": [0.15118699999999996, 0.246969, 1.233902, 1.1846420000000002, -1.0, 3.9731249999999996, 0.11476700000000006, -1.0, -1.0, 3.505625, 2.073133, -1.0, 3.727685, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.860694408416748, 0.09182599999999996, -1.0, -1.0, 2.280573, -1.0, 2.39541, -1.0, 0.10316799999999993, -1.0, -1.0, -1.0, 2.651597, -1.0, 0.09182599999999996, 2.950496, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, -0.9566131830215454, 1.7294070000000001, -1.0, -1.0, 4.550646, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 1.28551, 2.795066, 3.727685, 3.1194439999999997, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.150089979171753, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, 2.39541, -1.0, 1.890848, -1.0, -1.0, 1.3396370000000002, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -0.7447566986083984, "live_wins": 38, "live_total": 116, "age": 353, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.13284626286965723, "backtest_acc": 0.5688830943068232, "backtest_n": 2155}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding divergence 
    metrics that capture the relationship between funding, order flow, and the book.
    """
    df = df.copy()

    # 1. Divergence Metrics
    # Captures when funding and aggressive flow are in opposition (potential squeeze)
    df['funding_delta_divergence'] = np.sign(df['mean_funding']) * np.sign(df['sum_delta'])
    
    # Captures when aggressive flow and order book imbalance are in opposition (potential absorption)
    df['obi_delta_divergence'] = np.sign(df['mean_obi']) * np.sign(df['sum_delta'])
    
    # 2. Volatility-Adjusted Funding
    # Helps the model understand if funding is significant relative to current volatility
    df['funding_vol_ratio'] = df['mean_funding'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust, macro signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Squeeze & Absorption Congruence (SAC) Gating".

    Logic:
    - Identifies "Squeezes": High funding in one direction but aggressive delta in the other.
    - Identifies "Absorption": Aggressive delta in one direction but OBI in the other.
    - If the model's predicted side aligns with these pressure signals, we amplify the bet.
    - If the model's predicted side fights these signals, we heavily penalize or abstain.
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
    mean_funding = features_row.get('mean_funding', 0.0)
    sum_delta = features_row.get('sum_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)

    squeeze_mult = 1.0
    absorption_mult = 1.0

    # 1. Funding Squeeze Detection
    # If longs are paying (funding > 0) but selling is aggressive (delta < 0) -> Squeeze Down
    # If shorts are paying (funding < 0) but buying is aggressive (delta > 0) -> Squeeze Up
    if (mean_funding > 0 and sum_delta < 0):
        squeeze_dir = 'Down'
        squeeze_mult = 1.5 if side == squeeze_dir else 0.1
    elif (mean_funding < 0 and sum_delta > 0):
        squeeze_dir = 'Up'
        squeeze_mult = 1.5 if side == squeeze_dir else 0.1
    
    # 2. Order Flow Absorption Detection
    # If buying is aggressive (delta > 0) but the book is heavy on the sell side (obi < 0) -> Absorption Down
    # If selling is aggressive (delta < 0) but the book is heavy on the buy side (obi > 0) -> Absorption Up
    if (sum_delta > 0 and mean_obi < 0):
        absorption_dir = 'Down'
        absorption_mult = 1.3 if side == absorption_dir else 0.2
    elif (sum_delta < 0 and mean_obi > 0):
        absorption_dir = 'Up'
        absorption_mult = 1.3 if side == absorption_dir else 0.2

    # 3. Combine Multipliers and Calculate Effective EV
    effective_ev = best_ev * squeeze_mult * absorption_mult

    # 4. Time-Gated Hurdle
    # Increase required edge as the market approaches expiration to avoid late-stage noise
    # Threshold starts at ~2% and scales up to ~7% as seconds_remaining approaches 0
    threshold = 0.02 + (0.05 * (1.0 - (min(seconds_remaining, 300.0) / 300.0)))

    if effective_ev > threshold:
        # 5. Dynamic Sizing
        # Scale bet size by the effective edge, further amplified by the squeeze conviction
        bet_fraction = (effective_ev / 0.05) * squeeze_mult
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0