# Var 16 | BT: -0.155 | Live PnL: $-16.57 | Trades: 61
# Desc: Implements "Flow-Support Momentum Gating," which uses the alignment between price velocity and delta velocity to distinguish between "supported trends" and "absorption traps," applying a higher hurdle
# Stats: {"live_returns": [-1.0, -0.8544692397117615, -1.0, -1.0, 1.5191249999999998, 3.1194439999999997, 1.6554160000000002, -1.0, -0.7821702361106873, -1.0, -1.0, -1.0, -0.4331601858139038, -1.0, 1.890848, 1.5191249999999998, -1.0, -1.0, -0.3770897686481476, -1.0, -0.8170886039733887, -1.0, -1.0, -1.0, 0.05925399999999992, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7588549852371216, 0.13876599999999994, 0.4916779100894928, -1.0, 1.669612169265747, -0.8272225856781006, 3.1194439999999997, -1.0, 0.06021362543106079, -1.0, -1.0, -0.6733011603355408, -0.6979883313179016, -0.9773339629173279, 2.39541, 1.1678235530853271, -0.4348718225955963, -1.0, 0.7057457566261292, 1.3521380424499512, -1.0, -0.4782337248325348, -1.0, -0.7462236881256104, -1.0, -1.0, 1.9791409999999998, -0.7254714369773865, -0.45783454179763794, -0.8475821614265442, -0.5193099975585938], "live_pnl": -16.565019607543945, "live_wins": 16, "live_total": 61, "age": 72, "rounds_underwater": 52, "abstain_rounds": 0, "backtest_score": -0.15452836638457929, "backtest_acc": 0.5647155812036274, "backtest_n": 1996}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between price velocity and aggressive flow velocity.
    """
    # Work on a copy to avoid modifying the original dataframe
    df_work = df.copy()
    
    # 1. Calculate velocities (diffs) within each market window
    # We use binance_price as the primary liquidity reference
    df_work['delta_diff'] = df_work.groupby('market_slug')['sum_delta'].diff().fillna(0)
    df_work['price_diff'] = df_work.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Flow Alignment: Positive if price and delta move in the same direction (Trend)
    # Negative if price and delta move against each other (Absorption/Trap)
    df_work['flow_alignment'] = df_work['delta_diff'] * df_work['price_diff']

    # 3. Absorption Intensity: Magnitude of delta movement relative to price movement
    # High intensity means heavy delta is hitting the book but price isn't moving (Absorption)
    # We use a conditional to avoid division by zero
    df_work['absorption_intensity'] = np.where(
        df_work['price_diff'].abs() < 1e-7,
        df_work['delta_diff'].abs() * 100.0, 
        df_work['delta_diff'].abs() / df_work['price_diff'].abs()
    )
    # Cap intensity to prevent extreme outliers in XGBoost
    df_work['absorption_intensity'] = df_work['absorption_intensity'].clip(0, 100)

    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df_work.drop(columns=[c for c in cols_to_drop if c in df_work.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to combat 5-min noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Flow-Support Momentum Gating.
    The strategy requires a higher EV hurdle when the market shows signs of 
    absorption (price/delta divergence) and scales bet size down as absorption increases.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered microstructure metrics
    flow_alignment = features_row.get('flow_alignment', 0)
    absorption_intensity = features_row.get('absorption_intensity', 0)

    # 3. Regime-Based Hurdle
    # If flow and price are aligned (flow_alignment > 0), we are in a "Supported Trend"
    # If they are divergent, we are in an "Absorption" regime and need a much higher edge.
    if flow_alignment >= 0:
        hurdle = 0.025
    else:
        hurdle = 0.055

    # 4. Execution Logic
    if best_ev > hurdle:
        # Scaling factor 1: Absorption Penalty
        # Uses log-scaling to dampen the impact of extremely high absorption values
        absorption_penalty = 1.0 / (1.0 + 0.25 * math.log1p(absorption_intensity))

        # Scaling factor 2: Time Decay
        # Reduce exposure as the window closes to avoid last-second volatility spikes
        time_factor = max(0.2, seconds_remaining / 300.0)

        # Calculate final bet fraction
        # Scale linearly with edge relative to hurdle, then apply penalties
        bet_fraction = (best_ev / hurdle) * absorption_penalty * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure trades are meaningful
        if bet_fraction >= 0.1:
            return side, bet_fraction

    return None, 0.0