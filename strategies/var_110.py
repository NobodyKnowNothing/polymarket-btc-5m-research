# Var 110 | BT: 0.374 | Live PnL: $-1.58 | Trades: 143
# Desc: Implements "Potential-Kinetic Divergence (PKD) Gating," which distinguishes between momentum trends (congruent flow/book) and "coiled spring" setups (high-pressure order book vs. opposing aggressive f
# Stats: {"live_returns": [-1.0, 2.073133, 0.09182599999999996, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 2.795066, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -0.6238874793052673, 0.8781614303588867, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, -1.0, -1.0, 1.8077519999999998, 0.04885600000000001, -1.0, -1.0, -0.9226512312889099, -1.0, 3.505625, 0.04885600000000001, 1.28551, 1.7294070000000001, -1.0, 2.795066, 1.890848, 1.890848, -1.0, -0.7453691959381104, 0.16389999999999993, -1.0, 0.15118699999999996, 0.246969, 1.233902, 2.950496, 0.12663000000000002, -1.0, -1.0, -0.9934577941894531, -1.0, 3.9731249999999996, 0.6649420261383057, -1.0, -1.0, 0.10316799999999993, -1.0, -0.6622418761253357, 3.505625, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.8588359999999999, 0.8939649999999999, -0.966742217540741, 3.727685, -0.7219788432121277, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.950496, 0.9946252703666687, 3.9731249999999996, -1.0, -1.0, 1.5854279999999998, -1.0, 2.280573, 1.8077519999999998, -1.0, -1.0, 3.4412965774536133, -0.5950302481651306, 2.39541, -1.0, -1.0, -1.0, -1.0, -0.7396561503410339, 2.651597, -1.0, -1.0, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0628230571746826, -1.0, 0.05562227964401245, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, 0.03600156307220459, -1.0, -0.9537792205810547, -1.0, -0.9116910696029663, 1.3396370000000002, -1.0, -1.0, -1.0], "live_pnl": -1.5789109468460083, "live_wins": 52, "live_total": 143, "age": 255, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.3735120771838103, "backtest_acc": 0.5517470145953118, "backtest_n": 3315}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between latent liquidity (OBI) and kinetic flow (Delta).
    """
    df = df.copy()

    # 1. Congruence: Do the book and the flow agree? (1, 0, or -1)
    df['obi_delta_congruence'] = np.sign(df['mean_obi']) * np.sign(df['mean_delta'])

    # 2. Volatility-Normalized Delta: Is the flow significant relative to volatility?
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Pressure-Flow Ratio: How much 'unspent' order book pressure exists per unit of flow?
    # High values suggest a heavy book that isn't being actively cleared (potential spring).
    df['pressure_flow_ratio'] = df['mean_obi'] / (df['mean_delta'].abs() + 1e-6)

    # 4. Price Velocity: Local momentum
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Potential-Kinetic Divergence (PKD) Gating".

    Logic:
    - Trend Regime: OBI and Delta are aligned. High reliability, moderate threshold.
    - Spring Regime: OBI is high (heavy pressure) but Delta is opposing (absorption). 
      This is a 'coiled spring' setup. High reward, higher threshold.
    - Exhaustion Regime: OBI is neutral/low but Delta is high. Price is likely hitting a wall.
      High risk, very high threshold.
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
    congruence = features_row.get('obi_delta_congruence', 0.0)
    pressure = abs(features_row.get('mean_obi', 0.0))
    
    # 1. Identify the Microstructure Regime
    is_trend = congruence > 0
    is_divergent = congruence < 0
    
    # A "Spring" is a divergent regime where the book is heavily biased (high pressure)
    is_spring = is_divergent and pressure > 0.4
    # "Exhaustion" is a divergent regime where the book is thin/neutral (low pressure)
    is_exhaustion = is_divergent and pressure <= 0.4

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_trend:
        # Momentum: Follow the flow.
        req_ev = 0.03
        multiplier = 1.0
    elif is_spring:
        # Coiled Spring: Bet on the snap-back when the opposing flow fails.
        req_ev = 0.04
        multiplier = 1.2
    elif is_exhaustion:
        # Exhaustion: High risk of price stalling or reversing.
        req_ev = 0.055
        multiplier = 0.4
    else:
        # Neutral/Noise
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating: Tighten requirements as the window closes to avoid settlement noise
    if seconds_remaining < 60:
        req_ev *= 1.3
        multiplier *= 0.8

    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% target
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0