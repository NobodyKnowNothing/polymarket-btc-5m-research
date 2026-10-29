# Var 165 | BT: 1.174 | Live PnL: $-14.60 | Trades: 60
# Desc: Implements "Microstructural Absorption & Momentum (MAM)," which identifies high-probability reversals when aggressive flow hits limit-order walls (Absorption) and momentum breakouts when price and ord
# Stats: {"live_returns": [0.1812593638896942, -1.0, -0.2210599184036255, -1.0, -1.0, -0.9566118717193604, 1.1964099407196045, -1.0, -0.5698780417442322, -0.985668957233429, -0.204694002866745, -0.8174315094947815, -1.0, -0.8637589812278748, 0.49953410029411316, -0.8637589812278748, -0.2665458619594574, -0.43244802951812744, -0.16160908341407776, -1.0, 0.3800005316734314, -1.0, 0.026058748364448547, 0.04399383068084717, -0.204694002866745, -1.0, -1.0, -1.0, 1.283575415611267, -0.985668957233429, 3.1194439999999997, -0.9327676296234131, -1.0, -0.7004776000976562, -0.5158111453056335, -1.0, -1.0, 0.4529295861721039, -1.0, -0.34812918305397034, -0.7604192495346069, 0.5109384059906006, -0.21186475455760956, -0.2572202682495117, 4.245841, -1.0, 2.518758, -1.0, -1.0, -0.9034214615821838, 2.3827614784240723, 1.1846420000000002, -1.0, -0.48880043625831604, -0.350574791431427, -0.2124755084514618, -0.3485465347766876, -1.0, -0.6683076024055481, 0.6085849404335022], "live_pnl": -14.597914695739746, "live_wins": 15, "live_total": 60, "age": 83, "rounds_underwater": 59, "abstain_rounds": 0, "backtest_score": 1.1736200061738005, "backtest_acc": 0.5754012691302725, "backtest_n": 4105}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the divergence between aggressive market orders (Delta) and 
    passive limit order intent (OBI), as well as price-flow alignment.
    """
    df = df.copy()

    # 1. Price Velocity: Percentage change in price per snapshot (proxy for momentum)
    # Grouping by market_slug to prevent cross-market leakage
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].pct_change().fillna(0)

    # 2. Absorption Signal: High magnitude negative value indicates aggressive 
    # market orders are being absorbed by passive limit orders (e.g., high Delta, negative OBI).
    df['absorption_signal'] = df['sum_delta'] * df['mean_obi']

    # 3. Momentum Signal: Positive value indicates price movement is 
    # supported by order book imbalance (e.g., price up, OBI positive).
    df['momentum_signal'] = df['price_velocity'] * df['mean_obi']

    # 4. Volatility-Normalized Spread: Measures if price dispersion is 
    # excessive relative to current volatility (indicator of fragmentation/noise).
    df['vol_spread_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to combat the 50/50 noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructural Absorption & Momentum (MAM)".

    Logic:
    1. Detect Absorption (Reversal): If absorption_signal is strongly negative, 
       it means aggressive flow is hitting a wall. We bet AGAINST the flow.
    2. Detect Momentum (Trend): If momentum_signal is strongly positive, 
       it means price and OBI are in sync. We bet WITH the flow.
    3. Risk Mitigation: 
       - Scale down size if the vol_spread_ratio is high (market is fragmented).
       - Use a high hurdle for neutral regimes.
    """
    # Extract engineered features
    abs_sig = features_row.get('absorption_signal', 0.0)
    mom_sig = features_row.get('momentum_signal', 0.0)
    vol_spread = features_row.get('vol_spread_ratio', 1.0)
    sum_delta = features_row.get('sum_delta', 0.0)

    # Determine base EV
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Default: Follow the model's highest EV
    target_side = 'Up' if ev_up > ev_down else 'Down'
    best_ev = max(ev_up, ev_down)
    regime_multiplier = 0.6  # Default: Neutral/Low conviction

    # 1. Regime: Absorption Reversal
    # If abs_sig is very negative, Delta and OBI have opposite signs.
    # If Delta is positive (buying) but OBI is negative (bids being eaten), bet DOWN.
    if abs_sig < -2.0:
        regime_multiplier = 1.6
        target_side = 'Down' if sum_delta > 0 else 'Up'
    
    # 2. Regime: Momentum Breakout
    # If mom_sig is very positive, price and OBI are aligned.
    elif mom_sig > 0.001:
        regime_multiplier = 1.3
        target_side = 'Up' if sum_delta > 0 else 'Down'

    # Recalculate EV based on the regime-specific target side
    if target_side == 'Up':
        effective_ev = (prob_up - cost_up) * regime_multiplier
    else:
        effective_ev = (prob_down - cost_down) * regime_multiplier

    # 3. Execution
    if effective_ev > 0.02:
        # Scale bet size by edge, regime, and market stability
        # If vol_spread is high (> 1.5), reduce bet size by 50%
        stability_factor = 1.0 if vol_spread < 1.5 else 0.5
        
        # Target edge of 4% for full sizing
        bet_fraction = (effective_ev / 0.04) * regime_multiplier * stability_factor
        
        # Time decay: Reduce size as the window closes to avoid late-game volatility
        time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        
        return target_side, min(1.0, max(0.1, bet_fraction * time_factor))

    return None, 0.0