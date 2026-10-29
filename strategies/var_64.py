# Var 64 | BT: -0.123 | Live PnL: $-1.06 | Trades: 67
# Desc: Implements "Asymmetric Absorption & Flow-Inertia Gating (AAFIG)," which targets high-alpha reversal windows where aggressive market orders (Delta) are being absorbed by limit orders (OBI) without sign
# Stats: {"live_returns": [-0.9708038568496704, 0.6036400198936462, 0.02548438310623169, 0.04569143056869507, -1.0, -1.0, 3.727685, -0.254981130361557, -1.0, 0.7836606502532959, 0.9297810792922974, -1.0, -0.34628161787986755, -0.6577810645103455, 1.0720356702804565, -0.9213567972183228, -1.0, -1.0, -0.6569260954856873, -1.0, 0.33117175102233887, -1.0, 0.5858428478240967, -0.7652011513710022, -1.0, -1.0, 1.7294070000000001, -1.0, 0.4333306550979614, -1.0, -1.0, -1.0, -0.5487839579582214, -1.0, -0.4141349792480469, 3.727685, 2.795066, 1.501115322113037, 3.727685, 3.1194439999999997, -0.5417044758796692, 1.5282142162322998, -1.0, -0.9051003456115723, -0.6153435111045837, -1.0, -0.6889998912811279, -0.722951352596283, 2.280573, -0.771752655506134, -0.4126870632171631, -0.5771960616111755, 1.0115106105804443, -1.0, -1.0, 1.623528242111206, 2.061501979827881, -0.3686017096042633, -0.28498777747154236, -0.5451671481132507, -1.0, -0.29282671213150024, -0.3280235230922699, -0.4430673122406006, -0.26858341693878174, -1.0, -0.39677536487579346], "live_pnl": -1.055965781211853, "live_wins": 21, "live_total": 67, "age": 268, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.12338657903278333, "backtest_acc": 0.5538811487062838, "backtest_n": 1319}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure absorption (the divergence between aggressive flow and book pressure).
    """
    df = df.copy()

    # 1. Price Velocity
    df['price_vel_pct'] = df.groupby('market_slug')['binance_price'].pct_change().fillna(0)

    # 2. Absorption Index: High magnitude negative value indicates heavy selling hitting strong bids
    # (or heavy buying hitting strong asks), suggesting price absorption/reversal.
    df['absorption_index'] = df['mean_obi'] * df['mean_delta']

    # 3. Price Efficiency: How much price moves per unit of aggressive delta
    # Low efficiency = high absorption (Delta is high, but price movement is low)
    df['price_efficiency'] = df['price_vel_pct'].abs() / (df['mean_delta'].abs() + 1e-6)

    # 4. Delta Intensity
    df['delta_intensity'] = df['mean_delta'].abs()

    # 5. Volatility-Adjusted Flow
    df['flow_vol_ratio'] = df['delta_intensity'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on strong microstructure signals."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Asymmetric Absorption & Flow-Inertia Gating (AAFIG)".
    
    The strategy identifies 'Absorption' regimes:
    - Absorption Up: High Bids (OBI > 0.2) + Aggressive Selling (Delta < -0.2) -> Target 'Up'
    - Absorption Down: High Asks (OBI < -0.2) + Aggressive Buying (Delta > 0.2) -> Target 'Down'
    
    In these regimes, the hurdle is lowered because absorption is a high-conviction reversal signal.
    In normal regimes, a higher hurdle is required to ensure the edge is significant.
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # 2. Extract Microstructure State
    mean_obi = features_row.get('mean_obi', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    absorption_idx = features_row.get('absorption_index', 0.0)

    # 3. Identify Absorption Regime & Set Dynamic Hurdle
    # We look for divergence: OBI and Delta having opposite signs with high magnitude
    is_absorption_up = (mean_obi > 0.2) and (mean_delta < -0.2)
    is_absorption_down = (mean_obi < -0.2) and (mean_delta > 0.2)

    if is_absorption_up:
        target_side = 'Up'
        target_ev = ev_up
        hurdle = 0.02  # Lower hurdle for high-conviction absorption
    elif is_absorption_down:
        target_side = 'Down'
        target_ev = ev_down
        hurdle = 0.02
    else:
        # Standard regime: pick the best EV and use a higher hurdle
        if ev_up > ev_down:
            target_side = 'Up'
            target_ev = ev_up
        else:
            target_side = 'Down'
            target_ev = ev_down
        hurdle = 0.045 # Higher hurdle to filter noise in non-absorption regimes

    # 4. Execution Decision
    # Check if the best EV meets the regime-specific hurdle
    if target_ev < hurdle:
        return None, 0.0

    # Model Sanity Check: The model's probability must at least support the direction
    if target_side == 'Up' and prob_up < 0.5:
        return None, 0.0
    if target_side == 'Down' and prob_down < 0.5:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale by edge strength (EV/Hurdle)
    edge_ratio = target_ev / hurdle
    
    # Scale by absorption strength (if in absorption regime, increase size)
    absorption_multiplier = 1.0 + abs(absorption_idx)
    
    # Scale by time remaining (reduce exposure as window closes)
    time_factor = max(0.1, seconds_remaining / 300.0)

    # Final bet fraction calculation
    # 0.4 is a safety coefficient to keep bet sizes reasonable
    bet_fraction = edge_ratio * absorption_multiplier * time_factor * 0.4
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to avoid micro-trades
    if bet_fraction < 0.1:
        return None, 0.0

    return target_side, bet_fraction