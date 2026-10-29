# Var 171 | BT: 0.369 | Live PnL: $-0.26 | Trades: 113
# Desc: Implements "Liquidity-Efficiency Divergence (LED)," which differentiates between "Vacuum" regimes (low friction, high price impact) and "Absorption" regimes (high friction, low price impact) to switch
# Stats: {"live_returns": [-0.4, -0.4, 0.58249, 1.6983364000000005, -0.4, -0.4, 1.0075032, 0.4738568000000001, -0.4, -0.4, -0.4, -0.4, -0.4, 0.58249, 0.4935608, 0.9122292, 0.045906800000000025, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.6076499999999999, -0.4, -0.4, -0.4, -0.6763644218444824, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.05065200000000003, 1.2477776, -0.4, 4.95567, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.6341712, 1.6983364000000005, 0.6917628000000001, -0.4, 0.9581640000000001, -0.4, -0.4, 0.4550295999999999, -0.4, -0.4, -0.4, 0.4738568000000001, 0.9122292, 0.4550295999999999, 1.0075032, 0.9581640000000001, -0.4, -0.4, 0.6341712, -0.4, 0.7231008, 1.1801984, 0.5358548000000001, 0.6917628000000001, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0.8693576000000002, 1.1801984, -0.4, -0.4, -0.4, -0.4, 1.2477776, -0.4, 0.04126719999999995, 0.7916564, -0.4, 0.03229280000000001, -0.4, 0.437022, -0.4, 0.4935608, -0.4, -0.4, -0.4, -0.4, -0.4, 1.0606388, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4], "live_pnl": -0.2551382064819334, "live_wins": 36, "live_total": 113, "age": 277, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.3687467775352812, "backtest_acc": 0.549150036954915, "backtest_n": 2558}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'effort vs. result' metric (Price Impact Efficiency).
    """
    df = df.copy()

    # 1. Absorption Index: Ratio of aggressive flow to price movement.
    # High value = "The Wall" (High delta, little price move -> Absorption).
    # Low value = "The Vacuum" (Low delta, high price move -> Efficiency).
    # We use a small epsilon to prevent division by zero.
    df['absorption_index'] = df['mean_delta'].abs() / (df['price_vs_strike_pct'].abs() + 1e-5)

    # 2. Microstructure Accelerations: Rate of change in intent and flow.
    # Grouping by market_slug to prevent cross-market leakage.
    df['obi_accel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 3. Relative Volatility: Normalized delta intensity.
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

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
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Efficiency Divergence (LED)".

    Logic:
    1. Identify Regime via Absorption Index:
       - Vacuum Regime (Low Absorption): Price moves easily. 
         Strategy: Trend-following (Trust the ML model's direction).
       - Absorption Regime (High Absorption): Price is hitting a 'wall'. 
         Strategy: Mean-reversion (Trade against the aggressive flow if ML predicts reversal).
    2. Dynamic Sizing:
       - In Vacuum: Scale bet size by the efficiency of the move.
       - In Absorption: Use a smaller, fixed 'reversal' bet size.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    absorption = features_row.get('absorption_index', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    delta_sign = np.sign(mean_delta)
    
    side = None
    best_ev = 0.0
    
    # Thresholds for regime classification
    # Note: These are heuristic; the ML model provides the directional probability.
    VACUUM_THRESHOLD = 10.0
    ABSORPTION_THRESHOLD = 200.0

    # --- REGIME 1: VACUUM (High Efficiency) ---
    # Small amounts of delta are moving price significantly.
    # We follow the ML model's direction with higher conviction.
    if absorption < VACUUM_THRESHOLD:
        if ev_up > 0.03:
            side = 'Up'
            best_ev = ev_up
        elif ev_down > 0.03:
            side = 'Down'
            best_ev = ev_down
            
        if side:
            # Scale bet: Higher efficiency (lower absorption) -> higher conviction.
            # We use log1p to dampen the scaling effect.
            efficiency_multiplier = 1.0 / (1.0 + math.log1p(absorption))
            bet_fraction = (best_ev / 0.04) * efficiency_multiplier
            return side, min(1.0, max(0.1, bet_fraction))

    # --- REGIME 2: ABSORPTION (Low Efficiency) ---
    # Large amounts of delta are failing to move price.
    # We look for the ML model to signal a reversal of the current aggressive flow.
    elif absorption > ABSORPTION_THRESHOLD:
        # If flow is buying (delta > 0) but ML predicts 'Down' (ev_down is high)
        if delta_sign > 0 and ev_down > 0.03:
            side = 'Down'
            best_ev = ev_down
        # If flow is selling (delta < 0) but ML predicts 'Up' (ev_up is high)
        elif delta_sign < 0 and ev_up > 0.03:
            side = 'Up'
            best_ev = ev_up
            
        if side:
            # Cautious bet on the reversal.
            return side, 0.4

    # --- REGIME 3: NEUTRAL ---
    # Market is in a standard state; we abstain to protect Sharpe.
    return None, 0.0