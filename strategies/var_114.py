# Var 114 | BT: -1.054 | Live PnL: $-1.60 | Trades: 92
# Desc: Implements "Consensus-Validated Impulse (CVI)," which filters trades by requiring cross-exchange price convergence (lowering dispersion) alongside accelerating order book and flow momentum.
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, 3.505625, 0.03982502222061157, 1.28551, 1.1375739999999999, 1.7294070000000001, 0.1769179999999999, 2.795066, 1.890848, 1.890848, -1.0, -1.0, -1.0, -1.0, 0.15118699999999996, 0.246969, 1.233902, 2.950496, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, -1.0, -1.0, -1.0, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, -1.0, -0.7002966403961182, -1.0, -0.7964622378349304, 3.727685, -0.8295546770095825, -0.7444934844970703, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -0.850811243057251, -1.0, -1.0, 2.950496, 1.4562249999999999, 3.9731249999999996, -1.0, -0.9569317102432251, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -1.0, 4.893558, -1.0, -1.0, -1.0, 0.0700308084487915, -1.0, -1.0, -1.0, 2.651597, -1.0, 0.09182599999999996, 0.025201082229614258, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.6047077178955078, "live_wins": 32, "live_total": 92, "age": 164, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -1.053788503728818, "backtest_acc": 0.5561883899233296, "backtest_n": 3658}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the convergence of cross-exchange prices and the acceleration of order flow.
    """
    df = df.copy()

    # 1. Price Dispersion Velocity: Is the market reaching consensus?
    # Negative = exchanges are converging on a price (high conviction).
    # Positive = exchanges are diverging (high uncertainty/noise).
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 2. Momentum of Microstructure: Are the book and flow accelerating?
    # We look at the first derivative of OBI and Delta to find "ignition" phases.
    df['obi_momentum'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['delta_momentum'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 3. Volatility-Normalized Dispersion: How significant is the current spread?
    df['vol_normalized_spread'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to focus on structural signals."""
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
    Decide whether to trade using "Consensus-Validated Impulse (CVI)".

    Logic:
    - Consensus Breakout: Price spread is shrinking (convergence) AND 
      OBI momentum and Delta momentum are moving in the same direction (alignment).
      Requires low EV threshold (0.025).
    - Divergent Noise: Price spread is widening (divergence). 
      Requires very high EV threshold (0.05) to avoid fakeouts.
    - Stagnant/Neutral: Moderate EV threshold (0.035).
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
    spread_vel = features_row.get('spread_velocity', 0.0)
    obi_mom = features_row.get('obi_momentum', 0.0)
    delta_mom = features_row.get('delta_momentum', 0.0)

    # 1. Identify the Market Regime
    # Consensus: Price dispersion is decreasing
    is_consensus = spread_vel < 0
    
    # Alignment: Book and flow are accelerating in the same direction
    # (The product of their momentum is positive)
    is_aligned = (obi_mom * delta_mom) > 0

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_consensus and is_aligned:
        # High conviction: The market is converging and momentum is accelerating
        req_ev = 0.025
        multiplier = 1.0
    elif spread_vel > 0:
        # High risk: Exchanges are disagreeing on the price; likely a fakeout or high noise
        req_ev = 0.05
        multiplier = 0.5
    else:
        # Neutral: No clear consensus or momentum alignment
        req_ev = 0.035
        multiplier = 0.7

    # 3. Time-Gating
    # As the window closes, increase the hurdle to avoid late-stage volatility
    if seconds_remaining < 60:
        req_ev *= 1.2
        multiplier *= 0.8

    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% target
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, bet_fraction)

    return None, 0.0