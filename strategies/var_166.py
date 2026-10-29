# Var 166 | BT: 0.167 | Live PnL: $-17.72 | Trades: 60
# Desc: Implements "Pressure-Velocity & Liquidity-Quality Gating (PVLQ)," which uses the first derivative of the order-flow/imbalance interaction to identify momentum acceleration while scaling conviction by 
# Stats: {"live_returns": [-1.0, -1.0, -0.8547219634056091, -1.0, -1.0, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.28551, -1.0, 0.11296623945236206, 0.15118699999999996, -1.0, -1.0, -1.0, 1.890848, -1.0, 3.1194439999999997, -1.0, -0.6372440457344055, -1.0, -0.882412850856781, -1.0, -0.9902129769325256, -1.0, -0.6815346479415894, -1.0, 1.4562249999999999, -0.7117214798927307, -0.7675604820251465, -1.0, 4.245841, -1.0, 2.518758, -1.0, -1.0, 2.518758, -1.0, -1.0, -0.777501106262207, 0.246969, -1.0, 1.4562249999999999, -1.0, 2.280573, 0.11230432987213135, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0], "live_pnl": -17.715763092041016, "live_wins": 15, "live_total": 60, "age": 98, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 0.16697509864225385, "backtest_acc": 0.5583549462763987, "backtest_n": 3923}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    acceleration of order-flow pressure and the quality of available liquidity.
    """
    df = df.copy()

    # 1. Pressure Velocity: The rate of change of the OBI-Delta interaction.
    # A positive velocity means the 'pressure' is building; negative means it's exhausting.
    # We group by market_slug to ensure diffs are only calculated within the same window.
    df['pressure_velocity'] = df.groupby('market_slug')['obi_delta_interaction'].diff().fillna(0)

    # 2. Liquidity Quality: Inversely proportional to the OBI spread.
    # High spread = low liquidity = higher risk of slippage/reversals.
    df['liquidity_quality'] = 1.0 / (1.0 + df['obi_spread'])

    # 3. Volatility-Adjusted Pressure: Normalizing the interaction by current volatility.
    df['vol_adj_pressure'] = df['obi_delta_interaction'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Pressure-Velocity & Liquidity-Quality Gating (PVLQ)".

    Logic:
    1. Identify Microstructure Momentum:
       - ACCELERATION: The interaction (OBI * Delta) and its velocity have the same sign.
         (e.g., Pressure is positive and increasing). High conviction.
       - EXHAUSTION: The interaction and its velocity have opposite signs.
         (e.g., Pressure is positive but decreasing). Low conviction/Mean-reversion risk.
    2. Liquidity Scaling:
       - Scale the effective Expected Value (EV) by 'liquidity_quality' to avoid 
         trading in thin, high-spread environments.
    3. Execution:
       - Use a dynamic conviction multiplier to adjust the EV hurdle.
       - Scale bet size by edge, conviction, liquidity, and time-to-close.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    p_int = features_row.get('obi_delta_interaction', 0.0)
    v_int = features_row.get('pressure_velocity', 0.0)
    lq = features_row.get('liquidity_quality', 1.0)

    # 1. Determine Conviction Regime
    # If p_int and v_int have the same sign, pressure is accelerating.
    if p_int * v_int > 0:
        conviction = 1.4  # Acceleration regime
    # If they have opposite signs, pressure is slowing down (exhaustion).
    elif p_int * v_int < 0 and abs(p_int) > 0.01:
        conviction = 0.6  # Exhaustion regime
    else:
        conviction = 1.0  # Neutral

    # 2. Adjusted EV Calculation
    # We penalize the EV based on liquidity quality and the conviction regime.
    effective_ev = best_ev * conviction * lq

    # 3. Thresholding and Sizing
    # Base hurdle of 2.5% for the adjusted EV.
    if effective_ev > 0.025:
        # Time factor: reduce exposure as the market approaches the 5-minute close
        # to avoid high-variance "last-second" noise.
        time_factor = min(1.0, seconds_remaining / 120.0)

        # Bet sizing: Scale by (Edge / Target_Edge) * Conviction * Liquidity * Time
        # Target edge of 4% (0.04) for full sizing.
        bet_fraction = (best_ev / 0.04) * conviction * lq * time_factor

        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0