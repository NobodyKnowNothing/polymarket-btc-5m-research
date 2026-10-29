# Var 160 | BT: -0.502 | Live PnL: $-11.68 | Trades: 60
# Desc: Implements Regime-Specific Convexity Scaling (RSCS), which applies asymmetric EV hurdles and sizing based on whether the market is in a "Trend" regime (aligned Delta/OBI) or an "Absorption" regime (op
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 1.4562249999999999, 1.9791409999999998, 0.08160946518182755, 0.22335556149482727, -0.012052488513290882, 2.073133, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, 1.890848, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, 1.4562249999999999, 0.04885600000000001, 4.245841, -1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, -1.0, -1.0, -0.07608652859926224], "live_pnl": -11.679176330566406, "live_wins": 17, "live_total": 60, "age": 103, "rounds_underwater": 46, "abstain_rounds": 0, "backtest_score": -0.5024494753944914, "backtest_acc": 0.5444986856928277, "backtest_n": 3368}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the structural relationship between aggressive flow and book intent.
    """
    df = df.copy()

    # 1. Delta-OBI Alignment: +1 if they agree, -1 if they oppose, 0 if neutral.
    # This captures whether the market is trending or in an absorption battle.
    df['delta_obi_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 2. Absorption Ratio: Magnitude of aggressive flow relative to book resistance.
    # Only calculated when Delta and OBI are in opposition (the "battleground").
    # We use a +0.5 smoothing factor to prevent division by zero/extreme spikes.
    df['absorption_ratio'] = np.where(
        df['mean_delta'] * df['mean_obi'] < 0,
        abs(df['mean_delta']) / (abs(df['mean_obi']) + 0.5),
        0.0
    )

    # Drop metadata and target as per instructions
    # Note: seconds_remaining is passed separately to strategy_filter, 
    # so we drop it from the training matrix X.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Specific Convexity Scaling (RSCS)".

    Logic:
    1. Identify Regime:
       - Trend Regime: Delta and OBI are aligned (sign(delta) == sign(obi)).
       - Absorption Regime: Delta and OBI are opposing (sign(delta) != sign(obi)).
    2. Asymmetric Hurdle:
       - Trend Regime: Lower hurdle (2.5% base) because trends are more stable.
       - Absorption Regime: Higher hurdle (5.0% base) because absorption battles 
         are highly volatile and prone to whipsaws.
       - Both regimes increase the hurdle as seconds_remaining decreases (time decay).
    3. Convex Sizing:
       - In Trend regimes, bet size is standard.
       - In Absorption regimes, bet size scales with the 'absorption_ratio'—we only 
         bet heavily when the pressure of the absorption is massive (potential breakout).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    best_ev = max(ev_up, ev_down)
    side = 'Up' if ev_up > ev_down else 'Down'

    # Extract engineered features
    alignment = features_row.get('delta_obi_alignment', 0.0)
    absorption = features_row.get('absorption_ratio', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    # Time decay component (increases hurdle as market approaches close)
    time_penalty = (300 - seconds_remaining) / 300 * 0.03

    # 1. Determine Regime-Specific Hurdle and Scaling
    if alignment > 0:
        # TREND REGIME
        base_hurdle = 0.025 + (mean_vol * 0.05)
        scaling_factor = 1.0
    elif alignment < 0:
        # ABSORPTION REGIME
        base_hurdle = 0.05 + (mean_vol * 0.15)
        # Scale bet size by the intensity of the absorption (capped at 1.0)
        scaling_factor = min(1.0, absorption / 2.0)
    else:
        # NEUTRAL/NOISE REGIME
        base_hurdle = 0.04 + time_penalty
        scaling_factor = 0.5

    total_hurdle = base_hurdle + time_penalty

    # 2. Execution
    if best_ev > total_hurdle:
        # Scale bet fraction:
        # - Magnitude of edge relative to the hurdle.
        # - Regime-specific scaling (Trend vs Absorption).
        bet_fraction = (best_ev / total_hurdle) * scaling_factor
        return side, min(1.0, max(0.01, bet_fraction))

    return None, 0.0