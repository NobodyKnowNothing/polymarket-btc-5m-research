# Var 30 | BT: -0.327 | Live PnL: $-1.12 | Trades: 83
# Desc: Volatility-Scaled Mispricing Capture (VSMC) — This variant shifts focus from predicting direction to quantifying the "signal-to-noise" ratio of the market's mispricing, using a dynamic hurdle that pen
# Stats: {"live_returns": [-1.0, -0.5405426621437073, -0.9727570414543152, -1.0, 2.518758, 0.09182599999999996, -1.0, 0.08073200000000003, -1.0, 1.171281099319458, -0.8720514178276062, 2.173394, -1.0, -1.0, -0.7207930088043213, -1.0, 3.9731249999999996, -1.0, -0.9406070113182068, 1.7294070000000001, 2.39541, 1.890848, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, 0.046765685081481934, 0.10316799999999993, 0.7929560542106628, -0.5473816990852356, 2.280573, -0.49273252487182617, -1.0, -1.0, 1.6126389503479004, 1.7043178081512451, 1.5916836261749268, -1.0, -0.9157282114028931, 0.046263694763183594, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -0.9643731117248535, 1.1692347526550293, 1.890848, 0.028705999999999898, -1.0, -1.0, -1.0, 1.6554160000000002, 3.727685, 4.245841, -0.5942658185958862, -1.0, -1.0, 0.11476700000000006, -1.0, 1.28551, -1.0, -1.0, -0.8854922652244568, -1.0, -1.0, -1.0, -1.0, -1.0, -0.6212292313575745, -0.7749309539794922, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -0.7281709313392639, -1.0], "live_pnl": -1.1166422963142395, "live_wins": 29, "live_total": 83, "age": 142, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.3267008924198198, "backtest_acc": 0.5881642512077294, "backtest_n": 1639}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the synchronization of aggressive flow with order book imbalance 
    and volatility-normalized momentum.
    """
    df = df.copy()

    # 1. OBI Agreement: Measure how many exchanges agree on the sign of imbalance
    s1 = np.sign(df['binance_obi'])
    s2 = np.sign(df['coinbase_obi'])
    s3 = np.sign(df['kraken_obi'])
    df['obi_agreement'] = (
        (s1 == s2).astype(float) + 
        (s2 == s3).astype(float) + 
        (s1 == s3).astype(float)
    ) / 3.0

    # 2. Delta-OBI Synchronization: High when aggressive flow hits the side with imbalance
    df['delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 3. Volatility-Normalized Delta: Measures flow strength relative to current noise
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Velocity: Rate of change in the primary exchange price
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata but keep pm_costs for the model
    # seconds_remaining is used in strategy_filter, not as a feature for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent noise overfitting."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Scaled Mispricing Capture".
    The strategy targets high Expected Value (EV) relative to the current 
    volatility and price distance from the strike.
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

    # 2. Extract Engineered Metrics
    vol = features_row.get('mean_volatility', 0.01)
    dist = features_row.get('price_vs_strike_pct', 0.0)
    agreement = features_row.get('obi_agreement', 0.33)
    sync = features_row.get('delta_obi_sync', 0.0)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 2%. We increase it if:
    # - Volatility is high (noise risk)
    # - Price is far from strike (mean-reversion/exhaustion risk)
    # - OBI agreement is low (structural uncertainty)
    hurdle = 0.02 + (0.06 * vol) + (0.05 * abs(dist))
    
    if agreement < 0.7:
        hurdle += 0.02

    # 4. Execution Logic
    if best_ev > hurdle:
        # Time-decay: Scale down exposure as the window closes
        time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

        # Confidence Scaling:
        # We scale the bet by the ratio of edge to hurdle, 
        # and boost it if the delta and OBI are synchronized.
        sync_boost = 1.0
        if (side == 'Up' and sync > 0) or (side == 'Down' and sync < 0):
            sync_boost = 1.2
        elif (side == 'Up' and sync < 0) or (side == 'Down' and sync > 0):
            sync_boost = 0.7 # Penalty for flow-imbalance divergence

        bet_fraction = (best_ev / hurdle) * time_factor * sync_boost
        bet_fraction = min(1.0, max(0.0, bet_fraction))

        # Minimum threshold to avoid tiny, inefficient trades
        if bet_fraction < 0.05:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0