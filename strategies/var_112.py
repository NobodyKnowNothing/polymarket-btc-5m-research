# Var 112 | BT: 0.956 | Live PnL: $-1.87 | Trades: 275
# Desc: Implements "Liquidity-Velocity Divergence (LVD) Gating," which identifies "Liquidity Vacuums" (price moving while OBI thins) to apply extreme EV hurdles, preventing entries into unstable, high-reversa
# Stats: {"live_returns": [1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, 0.04885600000000001, 0.11476700000000006, -1.0, -1.0, 1.8077519999999998, 0.03982502222061157, -1.0, -1.0, 3.505625, 1.28551, 1.031211018562317, 1.7294070000000001, -1.0, 2.795066, 1.890848, 1.890848, -1.0, -1.0, 0.16389999999999993, -1.0, 0.15118699999999996, 0.246969, 1.218573808670044, 2.950496, 1.1846420000000002, -1.0, -1.0, -1.0, 0.10421991348266602, -1.0, -1.0, -1.0, 3.505625, 1.5191249999999998, 2.073133, -1.0, -1.0, -1.0, -0.8475574254989624, 4.893558, -1.0, -1.0, -1.0, -1.0, -0.8719606399536133, 6.238865, 0.08073200000000003, -1.0, -1.0, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.950496, 1.4562249999999999, -1.0, -1.0, 1.5854279999999998, -1.0, 2.280573, 1.8077519999999998, -0.9791890978813171, -1.0, -1.0, -0.9510998129844666, 10.723439, 4.893558, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.233902, -1.0, -1.0, 0.12663000000000002, 1.4562249999999999, -1.0, -1.0, -1.0, 0.11476700000000006, 1.890848, 2.651597, 0.09298592805862427, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, 0.09298592805862427, -1.0, -1.0, 0.08073200000000003, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -0.8859749436378479, 3.727685, 2.795066, 1.9791409999999998, -1.0, -1.0, 1.28551, 2.795066, 3.727685, 3.1194439999999997, 1.3396370000000002, -1.0, -1.0, 1.9791409999999998, -0.8437499403953552, -1.0, -1.0, -1.0, 3.303756, 0.06987699999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.08504647016525269, 2.795066, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, -0.7656365633010864, -1.0, -1.0, -1.0, 2.073133, 2.280573, -1.0, -1.0, -0.826209545135498, -1.0, -1.0, -1.0, 0.9987936019897461, -1.0, -0.8437499403953552, -1.0, 2.795066, -1.0, -1.0, 0.15118699999999996, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -0.969587504863739, 1.7294070000000001, -1.0, -1.0, -0.8079044818878174, -1.0, -1.0, 1.5191249999999998, -0.792374849319458, -1.0, -1.0, 2.39541, 1.28551, 1.1375739999999999, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, -1.0, 1.3396370000000002, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, 3.727685, -1.0, 1.3396370000000002, -1.0, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -0.9062785506248474, 0.08781695365905762, -1.0, 6.238865, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, 0.11335402727127075, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.866508960723877, "live_wins": 93, "live_total": 275, "age": 465, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.9557986536922032, "backtest_acc": 0.5572836801752464, "backtest_n": 3397}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the second-order relationship between price velocity and liquidity velocity.
    """
    df = df.copy()

    # 1. Price Velocity (Kinetic)
    # We use the difference in binance_price to capture immediate momentum
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Liquidity Velocity (Potential Change)
    # Captures whether the order book is thickening or thinning
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 3. Volatility-Normalized Velocity
    # Helps the model understand if a price move is significant relative to current noise
    df['vol_norm_velocity'] = df['price_velocity'] / (df['mean_volatility'] + 1e-6)

    # 4. Momentum-Liquidity Alignment
    # Do the price direction and the book direction match?
    df['vel_obi_alignment'] = np.sign(df['vol_norm_velocity']) * np.sign(df['mean_obi'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters optimized for microstructure signal detection."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Velocity Divergence (LVD) Gating".

    Logic:
    - Vacuum Regime: Price is moving fast (high vol_norm_velocity) but OBI is dropping (negative obi_velocity).
      This indicates a 'liquidity hole' where price is running into thin air. Extremely high reversal risk.
    - Wall Regime: Price is stagnant (low vol_norm_velocity) but OBI is high. 
      This indicates a heavy limit order wall. High potential for a breakout or absorption.
    - Trend Regime: Price and OBI are moving in tandem. Standard momentum.
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
    v_norm_vel = features_row.get('vol_norm_velocity', 0.0)
    obi_vel = features_row.get('obi_velocity', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)

    # 1. Identify the Microstructure Regime
    # A "Vacuum" is high velocity with thinning liquidity
    is_vacuum = abs(v_norm_vel) > 1.5 and obi_vel < -0.02
    
    # A "Wall" is low velocity with high order book pressure
    is_wall = abs(v_norm_vel) < 0.5 and abs(mean_obi) > 0.3
    
    # A "Trend" is velocity aligned with the book
    is_trend = abs(v_norm_vel) > 0.8 and np.sign(v_norm_vel) == np.sign(mean_obi)

    # 2. Set Regime-Specific Thresholds and Scaling
    if is_vacuum:
        # Vacuum: Extreme risk of immediate reversal. Only take massive edges.
        req_ev = 0.06
        multiplier = 0.5
    elif is_wall:
        # Wall: High conviction setup for a breakout or absorption.
        req_ev = 0.025
        multiplier = 1.2
    elif is_trend:
        # Trend: Standard momentum following.
        req_ev = 0.035
        multiplier = 1.0
    else:
        # Noise/Neutral
        req_ev = 0.04
        multiplier = 0.7

    # 3. Time-Gating: As the window closes, liquidity-based signals become more erratic
    if seconds_remaining < 90:
        req_ev *= 1.4
        multiplier *= 0.7

    if best_ev > req_ev:
        # Scale bet size by the magnitude of the edge relative to a 4% target
        # We cap the multiplier to prevent over-leveraging in high-vol regimes
        bet_fraction = multiplier * (best_ev / 0.04)
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0