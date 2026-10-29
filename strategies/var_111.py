# Var 111 | BT: -0.368 | Live PnL: $-1.36 | Trades: 64
# Desc: Implements "Lead-Venue Impulse & Global Consensus (LVIGC)," which distinguishes between "Unified Breakouts" (high efficiency + high cross-exchange consensus), "Absorption Traps" (low efficiency + high
# Stats: {"live_returns": [0.09182599999999996, -1.0, 0.03314614295959473, 2.073133, 0.09182599999999996, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, -1.0, -0.8087015151977539, -0.5475990772247314, 1.7294070000000001, -0.6944006085395813, -1.0, -0.7997613549232483, 0.9216495752334595, -0.9094287157058716, 0.12663000000000002, 0.03600156307220459, -1.0, -0.5358605980873108, -1.0, -1.0, -1.0, 0.04885600000000001, 0.11476700000000006, -1.0, -1.0, 1.8077519999999998, 0.04885600000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5090635418891907, 3.505625, 0.028446435928344727, -0.6944006085395813, 0.8168304562568665, 1.7294070000000001, -0.5606755614280701, -1.0, 2.795066, 1.890848, 1.890848, -1.0, -1.0, -1.0, 0.13480401039123535, 0.246969, 0.628134548664093, 2.950496, -1.0, 0.8477306365966797, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.362393856048584, "live_wins": 27, "live_total": 64, "age": 75, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.3679911653873586, "backtest_acc": 0.5609918087226035, "backtest_n": 3550}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coordination between lead-venue impulse and global exchange consensus.
    """
    df = df.copy()

    # 1. Calculate Cross-Exchange Consensus (Agreement on Delta and OBI)
    # A high consensus value means all exchanges are seeing similar flow/imbalance.
    delta_cols = ['binance_delta', 'coinbase_delta', 'kraken_delta']
    obi_cols = ['binance_obi', 'coinbase_obi', 'kraken_obi']
    
    df['delta_std'] = df[delta_cols].std(axis=1)
    df['obi_std'] = df[obi_cols].std(axis=1)
    
    # Consensus is the inverse of the combined dispersion
    df['consensus'] = 1.0 / (df['delta_std'] + df['obi_std'] + 1e-6)

    # 2. Calculate Lead-Venue Efficiency
    # We measure how much the price moves relative to the largest single-exchange delta.
    df['max_abs_delta'] = df[delta_cols].abs().max(axis=1)
    df['price_momentum'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    
    # Efficiency: Price movement per unit of maximum aggressive delta
    df['efficiency'] = df['price_momentum'] / (df['max_abs_delta'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to single-exchange noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Venue Impulse & Global Consensus (LVIGC)".

    Logic:
    - Unified Breakout: High efficiency + High consensus. (Follow the trend)
    - Absorption Trap: Low efficiency + High consensus. (Avoid or trade reversal)
    - Fragmented Noise: High efficiency + Low consensus. (Avoid - one exchange is leading, others lag)
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        eff = features_row.get('efficiency', 0.0)
    else:
        best_ev = ev_down
        side = 'Down'
        eff = -features_row.get('efficiency', 0.0) # Use absolute efficiency magnitude

    con = features_row.get('consensus', 0.0)
    abs_eff = abs(features_row.get('efficiency', 0.0))

    # 1. Define Regimes
    is_breakout = (abs_eff > 0.02) and (con > 0.5)
    is_absorption = (abs_eff < 0.01) and (con > 0.5)
    is_fragmented = (abs_eff > 0.02) and (con <= 0.5)

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_breakout:
        # High conviction: Unified movement across venues
        req_ev = 0.022
        multiplier = 1.0
    elif is_absorption:
        # High risk: Price is stuck despite high consensus (potential reversal or exhaustion)
        req_ev = 0.045
        multiplier = 0.4
    elif is_fragmented:
        # High risk: One exchange is moving, but others are not (noise/arbitrage gap)
        req_ev = 0.040
        multiplier = 0.5
    else:
        # Neutral/Noise
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating
    # As the window closes, increase the hurdle to avoid being caught in end-of-window volatility
    if seconds_remaining < 60:
        req_ev *= 1.25
        multiplier *= 0.8

    # 4. Final Execution
    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% target
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, bet_fraction)

    return None, 0.0