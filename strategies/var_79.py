# Var 79 | BT: 0.559 | Live PnL: $-0.22 | Trades: 79
# Desc: Implements "Congruence-Funding Divergence (CFD)," which uses multi-exchange delta agreement to identify high-conviction trends and optimizes bet sizing for funding-driven squeezes.
# Stats: {"live_returns": [-1.0, 1.5191249999999998, 0.16389999999999993, 2.173394, -0.7009726762771606, 2.073133, 1.126562237739563, -1.0, 1.5191249999999998, 0.16389999999999993, -1.0, 4.550646, -0.7373656034469604, -0.8579704761505127, 2.700063705444336, -1.0, -1.0, 1.7645650000000002, 0.9465638399124146, -1.0, 0.9852904081344604, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, -0.8016247749328613, -1.0, 1.472541093826294, -1.0, 2.795066, 2.617840528488159, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, 2.280573, -0.916361927986145, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, -1.0, -1.0, 1.7294070000000001, -1.0, -0.8257747292518616, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, 1.4562249999999999, 1.890848, 1.28551, -1.0, -1.0, -1.0, -1.0, 1.1375739999999999, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, 1.7294070000000001, -0.7385365962982178, -1.0, -1.0, -1.0], "live_pnl": -0.21896100044250488, "live_wins": 28, "live_total": 79, "age": 424, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.559422918331344, "backtest_acc": 0.5165310274669379, "backtest_n": 862}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    multi-exchange delta congruence and funding-flow divergence.
    """
    df = df.copy()

    # 1. Delta Congruence: Do all three exchanges see the same aggressive flow?
    # 1.0 = perfect agreement, 0.0 = total disagreement, -1.0 = perfect opposite agreement.
    df['delta_congruence'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    ) / 3.0

    # 2. Delta Dispersion: How much do the exchanges disagree on the magnitude of flow?
    df['delta_dispersion'] = (
        abs(df['binance_delta'] - df['coinbase_delta']) + 
        abs(df['coinbase_delta'] - df['kraken_delta']) + 
        abs(df['kraken_delta'] - df['binance_delta'])
    ) / 3.0

    # 3. Funding-Flow Alignment: Is the aggressive flow aligned with the perp funding?
    # Positive = Crowded Trend (Flow and Funding in same direction)
    # Negative = Squeeze Potential (Flow and Funding in opposite directions)
    df['funding_flow_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_funding'])

    # 4. Volatility-Adjusted Congruence
    df['vol_adj_congruence'] = df['delta_congruence'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to combat the 5-min noise."""
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
    Decide whether to trade using "Congruence-Funding Divergence (CFD)".

    Logic:
    - High Congruence Filter: Only trade if delta agreement across exchanges is high (|congruence| >= 0.66).
    - Regime 1: Squeeze (Alignment < 0). Aggressive flow is fighting funding. 
      High alpha potential. Lower EV threshold, higher bet multiplier.
    - Regime 2: Crowded Trend (Alignment > 0). Aggressive flow follows funding. 
      Lower alpha, higher risk of exhaustion. Higher EV threshold, lower bet multiplier.
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
    alignment = features_row.get('funding_flow_alignment', 0.0)
    dispersion = features_row.get('delta_dispersion', 0.0)

    # 1. Congruence Gate: Avoid trading when exchanges are in conflict
    if abs(congruence) < 0.66:
        return None, 0.0

    # 2. Identify Regime and set parameters
    # Alignment < 0 means flow is moving against the funding (e.g., buying while funding is negative)
    # This is a classic short squeeze or long liquidation setup.
    if alignment < 0:
        # Squeeze Regime: High conviction, high alpha
        req_ev = 0.020
        multiplier = 1.2
    else:
        # Crowded Trend Regime: Standard momentum, higher exhaustion risk
        req_ev = 0.035
        multiplier = 0.75

    # 3. Dispersion Penalty: If exchanges disagree on the magnitude of flow, reduce confidence
    # We use a soft penalty rather than a hard gate to allow for some noise.
    effective_ev = best_ev / (1.0 + dispersion)

    # 4. Time-Gating: Tighten requirements as the window closes to avoid late-stage noise
    if seconds_remaining < 60:
        req_ev *= 1.3
        multiplier *= 0.8

    # 5. Final Execution
    if effective_ev > req_ev:
        # Scale bet size based on the magnitude of the edge relative to a 4% target
        bet_fraction = multiplier * (effective_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0