# Var 108 | BT: -0.653 | Live PnL: $-7.37 | Trades: 61
# Desc: Implements "Microstructure Elasticity & Congruence Gating (MECG)," which differentiates between "Breakout" regimes (high price impact per unit of delta) and "Absorption" regimes (low price impact desp
# Stats: {"live_returns": [-0.816256046295166, -1.0, -1.0, -1.0, -0.7098689079284668, -0.5172989368438721, -1.0, 1.9791409999999998, -0.7277939915657043, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5558279156684875, -0.5460951328277588, -1.0, 0.9954040050506592, -1.0, 3.505625, -1.0, -1.0, 1.8077519999999998, 1.8709615468978882, -1.0, 1.7294070000000001, -1.0, 0.08526402711868286, -1.0, -1.0, -0.548570990562439, -1.0, 0.11267781257629395, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -0.6423706412315369, -1.0, -0.5804402232170105, 3.505625, 1.28551, 1.7294070000000001, -0.4934086203575134, -1.0, 1.8692041635513306, 1.890848, 1.890848, -1.0, -0.8261780738830566, -1.0, -1.0, 0.10874873399734497, -0.921400249004364, 1.0497422218322754, 2.950496, -0.48406046628952026, -1.0, -0.1780930608510971], "live_pnl": -7.373248100280762, "live_wins": 18, "live_total": 61, "age": 94, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.653233527148476, "backtest_acc": 0.5331251394155699, "backtest_n": 3084}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price impact efficiency (elasticity) and flow-book congruence.
    """
    df = df.copy()

    # 1. Price Velocity: Raw change in price
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Impact Efficiency (Elasticity): How much price moves per unit of aggressive delta.
    # High value = Breakout/Trend (price moves easily).
    # Low value = Absorption (price is stuck despite heavy delta).
    df['impact_efficiency'] = df['price_velocity'].abs() / (df['mean_delta'].abs() + 1e-6)

    # 3. Flow Congruence: Do OBI and Delta agree?
    # Positive = both pushing same way; Negative = fighting.
    df['flow_congruence'] = df['mean_obi'] * df['mean_delta']

    # 4. Volatility-Spread Tension: Measures market "tightness"
    df['vol_spread_tension'] = df['mean_volatility'] * df['price_spread']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Elasticity & Congruence Gating (MECG)".

    Logic:
    - Breakout Regime: High impact_efficiency + flow congruence. High reward, lower hurdle.
    - Absorption Regime: Low impact_efficiency (price stuck). High risk, massive hurdle.
    - Coiled Spring: Low volatility + high efficiency. Maximum conviction.
    - Misalignment Penalty: If predicted side contradicts Delta/OBI direction, increase hurdle.
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
    eff = features_row.get('impact_efficiency', 0.0)
    cong = features_row.get('flow_congruence', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    vol = features_row.get('mean_volatility', 0.0)

    # 1. Identify Microstructure Regimes
    is_absorption = eff < 0.0005  # Price is being absorbed by limit orders
    is_breakout = eff > 0.005     # Price is moving easily with flow
    is_coiled = vol < 0.0008      # Market is in a low-volatility squeeze
    
    # 2. Check Directional Alignment
    # We want the predicted side to match the sign of the aggressive flow (delta)
    is_aligned = False
    if side == 'Up' and delta > 0:
        is_aligned = True
    elif side == 'Down' and delta < 0:
        is_aligned = True

    # 3. Set Dynamic Hurdle and Multiplier
    # Base requirements
    req_ev = 0.040
    multiplier = 0.8

    if is_breakout:
        if is_coiled:
            # Perfect setup: Coiled spring breakout
            req_ev = 0.020
            multiplier = 1.4
        else:
            # Standard breakout
            req_ev = 0.030
            multiplier = 1.1
    elif is_absorption:
        # Dangerous: Delta is high but price isn't moving. High risk of reversal/trap.
        req_ev = 0.055
        multiplier = 0.4
    
    # 4. Apply Penalties
    # If the model predicts 'Up' but Delta is negative, it's a contrarian/divergent trade.
    if not is_aligned:
        req_ev += 0.025
        multiplier *= 0.6

    # If OBI and Delta are fighting (low congruence), increase hurdle
    if cong < 0:
        req_ev += 0.015

    # 5. Execution
    if best_ev > req_ev:
        # Time decay: reduce exposure as the window closes to avoid end-of-window noise
        time_decay = min(1.0, seconds_remaining / 300.0)
        
        # Scale bet size by edge magnitude relative to a 4% baseline
        bet_fraction = multiplier * (best_ev / 0.04) * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0