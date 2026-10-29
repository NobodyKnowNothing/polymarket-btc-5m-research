# Var 78 | BT: -0.732 | Live PnL: $-2.04 | Trades: 120
# Desc: Implements "Tri-Force Congruence & Compression Scaling (TFCCS)," which identifies high-conviction breakouts by requiring alignment between order flow (Delta), order book (OBI), and price action (Momen
# Stats: {"live_returns": [1.6554160000000002, 0.05925399999999992, 1.9791409999999998, -1.0, -1.0, -1.0, -0.9800872802734375, 4.245841, -1.0, 1.5191249999999998, -1.0, -0.588210940361023, -1.0, -1.0, 1.9791409999999998, 2.173394, 2.073133, -1.0, -1.0, -0.8446269631385803, 1.233902, -1.0, -1.0, 0.04885600000000001, 1.2375576496124268, 2.073133, -0.5018270015716553, -1.0, -1.0, 3.303756, -1.0, 1.5854279999999998, 0.9375977516174316, 0.11708939075469971, 0.6299560070037842, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, -1.0, -1.0, 2.39541, -0.6775224208831787, -1.0, -1.0, 3.505625, 0.8247790336608887, 2.950496, 3.505625, 2.173394, -1.0, 1.890848, -1.0, -1.0, -1.0, 2.280573, 1.28551, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, 1.5854279999999998, -0.8401638865470886, -1.0, -1.0, 3.9731249999999996, 2.950496, -0.483356237411499, -1.0, -1.0, -0.8932251334190369, -1.0, -1.0, -1.0, -0.8287336230278015, -0.6203383207321167, -1.0, 0.7653086185455322, -0.9690378308296204, -1.0, 1.8077519999999998, -0.8932251334190369, -1.0, -0.6768195629119873, -1.0, -1.0, 1.28551, -0.7990894317626953, -1.0, -1.0, -1.0, -1.0, -0.5737096667289734, -1.0, -1.0, -0.515709400177002, 2.651597, 7.543445, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5147473812103271, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -2.0350141525268555, "live_wins": 37, "live_total": 120, "age": 174, "rounds_underwater": 3, "abstain_rounds": 0, "backtest_score": -0.7316721902913026, "backtest_acc": 0.5445039530731957, "backtest_n": 3137}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronization of the three primary market drivers: 
    Flow (Delta), Liquidity (OBI), and Price (Momentum).
    """
    df = df.copy()

    # 1. Tri-Force Congruence: Interaction between Delta, OBI, and Momentum
    # A positive value indicates all three are pushing in the same direction.
    df['delta_obi_prod'] = df['mean_delta'] * df['mean_obi']
    df['momentum_delta_prod'] = df['mean_momentum'] * df['mean_delta']
    df['tri_force_congruence'] = df['mean_delta'] * df['mean_obi'] * df['mean_momentum']

    # 2. Volatility-Adjusted Flow Intensity
    # How much delta is being applied per unit of volatility?
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price Velocity & Acceleration
    # Captures the immediate change in price within the market window
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    
    # 4. Dispersion/Compression Metric
    # Low dispersion often precedes explosive moves (coiled state)
    df['dispersion_index'] = df['price_std'] / (df['binance_price'] + 1e-6)

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
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Force Congruence & Compression Scaling (TFCCS)".

    Logic:
    - The "Tri-Force" is the alignment of Delta, OBI, and Momentum.
    - "Coiled" Regime: Low volatility + Low price dispersion.
    - "Breakout" (High Conviction): Coiled regime + High Tri-Force Congruence.
    - "Trend" (Moderate Conviction): High volatility + High Tri-Force Congruence.
    - "Noise/Divergence": Low Tri-Force Congruence (requires massive EV to trade).
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
    congruence = features_row.get('tri_force_congruence', 0.0)
    vol = features_row.get('mean_volatility', 0.0)
    dispersion = features_row.get('dispersion_index', 0.0)
    
    # Determine if the market is "coiled" (low volatility and low dispersion)
    # Note: Thresholds are heuristic; in a real system, these could be rolling z-scores.
    is_coiled = vol < 0.0015 and dispersion < 0.0005
    
    # Determine if the Tri-Force is aligned (all components have same sign)
    # We use the product of the components as a proxy for alignment strength
    is_congruent = abs(congruence) > 0.001 

    # 1. Define Regime-Based Parameters
    if is_coiled and is_congruent:
        # BREAKOUT REGIME: High conviction in explosive moves
        req_ev = 0.020
        multiplier = 1.2
    elif not is_coiled and is_congruent:
        # TREND REGIME: Steady momentum in a volatile market
        req_ev = 0.035
        multiplier = 0.8
    elif not is_congruent:
        # NOISE/DIVERGENCE REGIME: Conflicting signals (e.g., high delta but negative OBI)
        req_ev = 0.055
        multiplier = 0.4
    else:
        # NEUTRAL REGIME
        req_ev = 0.040
        multiplier = 0.6

    # 2. Time-Gating: As the market nears resolution, tighten the requirements
    # to avoid being caught in "end-of-window" noise/reversals.
    if seconds_remaining < 45:
        req_ev *= 1.3
        multiplier *= 0.7

    # 3. Final Decision and Dynamic Sizing
    if best_ev > req_ev:
        # Scale bet size by how much the EV exceeds the requirement, capped at 1.0
        # We use a target edge of 0.05 for the scaling denominator
        bet_fraction = multiplier * (best_ev / 0.05)
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0