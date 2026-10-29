# Var 42 | BT: 0.521 | Live PnL: $-16.77 | Trades: 61
# Desc: Implements "Liquidity-Vacuum Squeeze Capture (LVSC)," which targets explosive price movements by identifying the intersection of funding-driven aggressive flow (squeezes) and low order-book resistance
# Stats: {"live_returns": [-0.7454735636711121, 0.017418473958969116, -1.0, -1.0, 0.865764319896698, -1.0, -1.0, -0.6665372848510742, 0.048996031284332275, 1.7294070000000001, -0.6875782608985901, 0.04261338710784912, -0.970678985118866, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, -0.6082565188407898, -1.0, -0.2546067535877228, -1.0, -0.4891180396080017, 0.06546103954315186, -0.6828493475914001, -0.672459602355957, -1.0, -1.0, -0.5773629546165466, 0.13876599999999994, 0.22062000632286072, -0.740829586982727, 4.550646, 3.1194439999999997, -0.5098736882209778, -1.0, -1.0, -0.3461543917655945, -0.4049345850944519, -0.21207663416862488, -0.8924161195755005, 2.39541, 0.02089792490005493, 0.5438051223754883, -0.8329020738601685, -1.0, 0.9715338945388794, -0.8832951188087463, -1.0, -0.7814039587974548, -0.9102452993392944, -0.4506799876689911, -1.0, -0.7815061211585999, 1.9791409999999998, -0.7108007669448853, -1.0, -0.7056456804275513, -0.2273382544517517, -0.32441991567611694], "live_pnl": -16.774089813232422, "live_wins": 16, "live_total": 61, "age": 76, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": 0.5212061518780049, "backtest_acc": 0.5449814126394052, "backtest_n": 2349}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the confluence of funding-driven pressure and liquidity gaps.
    """
    X = df.copy()

    # 1. Funding-Delta Sync: Measures if aggressive flow is aligned with perp funding pressure.
    # High absolute value indicates a potential squeeze/short-covering move.
    X['funding_delta_sync'] = X['mean_delta'] * X['mean_funding']

    # 2. Spread Velocity: The rate of change in cross-exchange price dispersion.
    # Rapidly increasing spreads often precede high-volatility "vacuum" moves.
    X['spread_velocity'] = X.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 3. OBI Density: The magnitude of book imbalance.
    # Low density (near 0) indicates a "liquidity vacuum" where price can move easily.
    X['obi_density'] = X['mean_obi'].abs()

    # 4. Delta Acceleration: Change in aggressive trade flow.
    X['delta_accel'] = X.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 5. Price-Impact Velocity: How much price dispersion is changing relative to flow.
    X['spread_delta_ratio'] = X['spread_velocity'] / (X['mean_delta'].abs() + 1e-6)

    # Drop metadata and target columns
    # We keep pm_cost_up/down as features to help the model understand market pricing
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup for XGBoost
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on high-conviction signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Vacuum Squeeze Capture (LVSC)".
    This strategy seeks to exploit moments when a squeeze (Delta + Funding) 
    hits a liquidity vacuum (Low OBI + High Spread).
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract Microstructure Metrics
    funding_delta_sync = features_row.get('funding_delta_sync', 0.0)
    obi_density = features_row.get('obi_density', 0.0)
    price_spread = features_row.get('price_spread', 0.0)
    spread_velocity = features_row.get('spread_velocity', 0.0)

    # 3. Calculate Squeeze Intensity
    # High absolute value means delta and funding are pushing in the same direction.
    squeeze_intensity = abs(funding_delta_sync)

    # 4. Calculate Vacuum Factor
    # A vacuum exists if OBI is low (no resistance) and spread is expanding (price searching).
    # We use a small epsilon to prevent division by zero.
    vacuum_factor = (1.0 / (obi_density + 0.2)) * (1.0 + (abs(spread_velocity) / (price_spread + 0.01)))

    # 5. Total Conviction Multiplier
    # We amplify the signal when the squeeze aligns with a liquidity vacuum.
    # We use a logarithmic scale for the vacuum factor to prevent extreme outliers.
    conviction_multiplier = (1.0 + squeeze_intensity) * math.log1p(vacuum_factor)

    # 6. Execution & Sizing
    # Minimum EV hurdle adjusted by conviction: if conviction is high, we can take slightly lower EV.
    # If conviction is low, we require a much higher EV.
    base_hurdle = 0.03
    adjusted_hurdle = base_hurdle / (0.5 + 0.5 * conviction_multiplier)

    if best_ev < adjusted_hurdle:
        return None, 0.0

    # Time-decay: Reduce exposure as the 5-minute window approaches zero.
    time_factor = max(0.1, seconds_remaining / 300.0)

    # Final bet fraction: 
    # Scale by edge (best_ev), conviction (multiplier), and time.
    # We cap the multiplier to ensure we don't over-leverage on single-snapshot spikes.
    bet_fraction = (best_ev / 0.05) * min(1.2, conviction_multiplier / 2.0) * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum trade threshold
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction