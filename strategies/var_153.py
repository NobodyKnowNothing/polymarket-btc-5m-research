# Var 153 | BT: 0.472 | Live PnL: $-3.46 | Trades: 60
# Desc: Implements "Liquidity-State Regime Switching (LSRS)," which categorizes the market into three distinct microstructural states—**Squeeze** (aggressive flow meets heavy limit orders), **Sweep** (aggress
# Stats: {"live_returns": [-1.0, 1.049452, 0.05925399999999992, 0.08393847942352295, 0.06987699999999997, 2.651597, -1.0, 0.04885600000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.8568074107170105, 1.5191249999999998, 0.12663000000000002, 2.651597, -1.0, -1.0, 1.3396370000000002, 0.8568074107170105, -0.923516035079956, -1.0, -1.0, -1.0, 0.03600156307220459, 2.073133, -1.0, 0.11395108699798584, 1.4562249999999999, 1.9791409999999998, 1.233902, 1.9791409999999998, -1.0, 1.1846420000000002, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, 1.092555, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, 0.12663000000000002, 0.15118699999999996, -0.8378826379776001, -1.0], "live_pnl": -3.458444595336914, "live_wins": 27, "live_total": 60, "age": 92, "rounds_underwater": 6, "abstain_rounds": 0, "backtest_score": 0.4721484498678723, "backtest_acc": 0.5591743987881083, "backtest_n": 4061}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement relative to order book intent.
    """
    df = df.copy()

    # 1. Delta-to-OBI Ratio: Measures how much aggressive flow exists relative to book imbalance.
    # High ratio = Sweep (low liquidity); Low ratio = Absorption (high liquidity).
    df['delta_obi_ratio'] = df['mean_delta'] / (df['mean_obi'].abs() + 0.1)

    # 2. Price Impact Proxy: Measures how much delta is required to move the price.
    # High value = "Slippery" market (low impact); Low value = "Sticky" market (high impact).
    df['price_impact_proxy'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 3. Funding-Delta Divergence: Captures potential squeeze setups.
    df['funding_delta_divergence'] = df['mean_funding'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-State Regime Switching (LSRS)".

    Logic:
    1. Identify Regime:
       - SQUEEZE: Aggressive delta and OBI have opposite signs (e.g., selling into heavy bids).
         This is a high-conviction setup for a reversal/squeeze.
       - SWEEP: Delta is significantly larger than OBI (ratio > 2.0). 
         The market is thin; momentum is likely to continue.
       - ABSORPTION: Neither a squeeze nor a sweep. The market is in a standard state.

    2. Apply Regime-Specific Thresholds & Scaling:
       - SQUEEZE: Low EV hurdle (0.02), High bet multiplier (1.3x).
       - SWEEP: Moderate EV hurdle (0.03), Standard bet multiplier (1.0x).
       - ABSORPTION: High EV hurdle (0.05), Reduced bet multiplier (0.7x).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract features
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    ratio = features_row.get('delta_obi_ratio', 0.0)
    
    delta_sign = np.sign(mean_delta)
    obi_sign = np.sign(mean_obi)
    
    # Determine Regime
    is_squeeze = (delta_sign * obi_sign < 0) and (abs(ratio) < 1.5) and (abs(mean_delta) > 0.001)
    is_sweep = abs(ratio) > 2.0
    
    if is_squeeze:
        threshold = 0.02
        bet_multiplier = 1.3
    elif is_sweep:
        threshold = 0.03
        bet_multiplier = 1.0
    else:
        threshold = 0.05
        bet_multiplier = 0.7

    # Selection
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    if best_ev > threshold:
        # Scale bet fraction:
        # Base scaling: (EV / 0.04)
        # Apply regime multiplier
        # Apply time decay (be more cautious in the final 30 seconds)
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = (best_ev / 0.04) * bet_multiplier * time_factor
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0