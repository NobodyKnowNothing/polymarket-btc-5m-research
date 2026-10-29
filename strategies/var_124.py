# Var 124 | BT: 0.372 | Live PnL: $-1.79 | Trades: 60
# Desc: Targets "Liquidity-Impact Drivers" by identifying when multiple exchanges show congruent aggressive flow that is disproportionately moving the price relative to order book depth, filtering out outlier
# Stats: {"live_returns": [-1.0, 1.0081479999999998, -1.0, 1.4562249999999999, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, 0.038675000000000015, 4.550646, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 2.795066, 3.727685, 1.3396370000000002, 0.968532, -1.0, 0.06987699999999997, -1.0, -1.0, 1.049452, -1.0, -1.0, -1.0, -1.0, -0.734271764755249, 0.022797465324401855, 0.09182599999999996, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.1375739999999999, 1.1375739999999999, 1.9791409999999998, -1.0, -1.0, 1.4562249999999999, 1.890848, 1.28551, -1.0], "live_pnl": -1.7943388223648071, "live_wins": 24, "live_total": 60, "age": 193, "rounds_underwater": 21, "abstain_rounds": 1, "backtest_score": 0.37216422735220506, "backtest_acc": 0.5258548790658882, "backtest_n": 1428}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    exchange-specific liquidity impact and flow congruence.
    """
    df = df.copy()

    # 1. Impact Factor: How much 'force' is being applied per unit of OBI?
    # High impact means aggressive trades are moving through thin books.
    # We add a small constant to prevent division by zero.
    df['binance_impact'] = abs(df['binance_delta']) / (abs(df['binance_obi']) + 0.1)
    df['coinbase_impact'] = abs(df['coinbase_delta']) / (abs(df['coinbase_obi']) + 0.1)
    df['kraken_impact'] = abs(df['kraken_delta']) / (abs(df['kraken_obi']) + 0.1)

    # 2. Impact Aggregates
    df['max_impact'] = df[['binance_impact', 'coinbase_impact', 'kraken_impact']].max(axis=1)
    df['mean_impact'] = df[['binance_impact', 'coinbase_impact', 'kraken_impact']].mean(axis=1)
    
    # 3. Impact Ratio: Is one exchange driving the market disproportionately?
    # A high ratio suggests an outlier/noise; a low ratio suggests a broad market move.
    df['impact_ratio'] = df['max_impact'] / (df['mean_impact'] + 1e-6)

    # 4. Delta Congruence: Do all exchanges agree on the direction of aggressive flow?
    # 1.0 = perfect agreement, 0.0 = neutral/mixed, -1.0 = perfect opposite.
    df['delta_congruence'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    ) / 3.0

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to impact noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.015,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Impact Driver Consensus (LIDC)".

    Logic:
    - Congruence Gate: Only trade if there is strong directional agreement (|congruence| >= 0.66).
    - Outlier Gate: If one exchange's impact is vastly higher than others (impact_ratio > 3.0),
      it's likely a single-exchange anomaly; abstain.
    - Impact-Weighted EV: Scale the expected value by the log of the maximum impact. 
      High-impact moves (thin books) provide higher alpha.
    - Time-Gating: Increase the required EV threshold as the market window closes.
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
    congruence = features_row.get('delta_congruence', 0.0)
    impact_ratio = features_row.get('impact_ratio', 1.0)
    max_impact = features_row.get('max_impact', 0.0)

    # 1. Congruence Gate: Avoid trading when exchanges are in conflict
    if abs(congruence) < 0.66:
        return None, 0.0

    # 2. Outlier Gate: Avoid single-exchange spikes (low reliability)
    if impact_ratio > 3.5:
        return None, 0.0

    # 3. Impact-Weighted Edge: 
    # We amplify the perceived edge if the impact is high (moving price through thin liquidity).
    # Using log1p to dampen the multiplier effect.
    effective_ev = best_ev * (1.0 + math.log1p(max_impact))

    # 4. Dynamic Thresholding
    # Base requirement for effective EV
    req_ev = 0.025
    
    # Tighten requirements in the final minute to avoid late-window volatility noise
    if seconds_remaining < 60:
        req_ev += 0.015

    # 5. Execution
    if effective_ev > req_ev:
        # Scale bet size based on the magnitude of the effective edge relative to a 5% target
        bet_fraction = min(1.0, effective_ev / 0.05)
        return side, max(0.0, bet_fraction)

    return None, 0.0