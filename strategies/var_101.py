# Var 101 | BT: -1.673 | Live PnL: $-0.60 | Trades: 68
# Desc: Implements "Funding-Flow Tension & Impact-Efficiency Gating (FFT-IEG)," which differentiates between high-conviction trend-following (where funding and aggressive flow align) and liquidity-driven brea
# Stats: {"live_returns": [-1.0, -0.716788113117218, -0.2503957450389862, -0.44640064239501953, 3.1194439999999997, -0.5184528231620789, 2.2500412464141846, -0.3421327769756317, 0.34082382917404175, -0.19942745566368103, -0.6168363690376282, -0.5691982507705688, -0.2321263998746872, -0.9500563740730286, -1.0, 1.5528368949890137, -0.2652352750301361, -0.19562362134456635, -1.0, -0.381723016500473, -0.4803807735443115, 4.550646, -0.26747825741767883, 2.5631847381591797, -0.34479784965515137, -0.8537750840187073, -0.9987213015556335, -0.3814336657524109, -0.2299899309873581, -0.3697337508201599, 1.665755271911621, 0.28881174325942993, -0.25860732793807983, -0.7025904059410095, 0.995353102684021, -1.0, -0.25775575637817383, -0.6936319470405579, -0.6274514198303223, -0.9578045010566711, 1.1102068424224854, -0.28479939699172974, -1.0, -1.0, -0.5268675088882446, -0.31049904227256775, -1.0, -1.0, -0.3627302646636963, -0.3985655903816223, -0.3396042585372925, -0.23348075151443481, 1.4524836540222168, -0.5603987574577332, 3.505625, -0.4591020941734314, -0.6281659603118896, 0.4132174253463745, 0.9793190956115723, 2.795066, -1.0, -0.8596588373184204, -0.21376977860927582, 0.7549278736114502, -0.39492154121398926, -0.5774216651916504, -0.4395826458930969, -0.24032042920589447], "live_pnl": -0.6006951928138733, "live_wins": 16, "live_total": 68, "age": 119, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -1.6730927194382232, "backtest_acc": 0.5764598954783003, "backtest_n": 2853}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the alignment between funding rates and aggressive flow, and the 
    efficiency of price movement relative to delta.
    """
    df = df.copy()

    # 1. Funding-Flow Tension: Measures if aggressive trades are aligned with 
    # the perp funding direction. High absolute value = strong trend conviction.
    # (Positive funding + Positive delta = Longs paying and buying = Strong Bullish)
    df['funding_flow_tension'] = df['mean_funding'] * df['mean_delta']

    # 2. Impact Efficiency: How much price moves per unit of aggressive delta.
    # High value = price is moving easily (liquidity vacuum/breakout).
    # Low value = delta is being absorbed by the book (high resistance).
    df['impact_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 3. Price Velocity: The raw directional movement of the lead exchange.
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 4. Momentum-Delta Alignment: Are the recent candles moving in the 
    # same direction as the aggressive flow?
    df['momentum_delta_alignment'] = df['mean_momentum'] * df['mean_delta']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on structural signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Flow Tension & Impact-Efficiency Gating (FFT-IEG)".

    Logic:
    - Tension Regime: Funding and Delta are aligned. High conviction trend.
    - Breakout Regime: Low tension but high impact efficiency. Liquidity-driven move.
    - Absorption Regime: Low tension and low impact efficiency. High noise/resistance.
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
    tension = features_row.get('funding_flow_tension', 0.0)
    impact = features_row.get('impact_efficiency', 0.0)
    alignment = features_row.get('momentum_delta_alignment', 0.0)
    
    # 1. Identify Microstructure Regimes
    # Tension: Funding and Flow are working together (Strong Trend)
    is_tension_regime = abs(tension) > 0.0005 
    
    # Breakout: Price is moving very efficiently despite lack of funding support
    is_breakout_regime = abs(impact) > 5.0 and not is_tension_regime
    
    # Absorption/Noise: Everything is low or conflicting
    is_noise_regime = not is_tension_regime and not is_breakout_regime

    # 2. Set Regime-Specific Thresholds and Multipliers
    if is_tension_regime:
        # Strong trend alignment: lower entry barrier, higher conviction
        req_ev = 0.022
        multiplier = 1.2
    elif is_breakout_regime:
        # Liquidity breakout: moderate barrier, moderate conviction
        req_ev = 0.035
        multiplier = 1.0
    else:
        # Noise/Absorption: high barrier, low conviction
        req_ev = 0.050
        multiplier = 0.5

    # 3. Apply Momentum Alignment Check
    # If the model wants to go Up, but momentum/delta alignment is negative, penalize.
    if side == 'Up' and alignment < 0:
        req_ev += 0.015
        multiplier *= 0.7
    elif side == 'Down' and alignment > 0:
        req_ev += 0.015
        multiplier *= 0.7

    # 4. Time Gating: Reduce exposure as the window closes to avoid end-of-period volatility
    time_decay = min(1.0, seconds_remaining / 300.0)

    if best_ev > req_ev:
        # Scale bet size by edge magnitude, regime multiplier, and time decay
        # Baseline: 0.05 EV = 1.0 multiplier
        bet_fraction = multiplier * (best_ev / 0.05) * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0