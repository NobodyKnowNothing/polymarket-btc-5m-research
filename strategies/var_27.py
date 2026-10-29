# Var 27 | BT: -0.192 | Live PnL: $-32.55 | Trades: 61
# Desc: Implements "Volatility-Adaptive Consensus Gating (VACG)," which uses a dynamic hurdle threshold that scales with market volatility, exchange OBI disagreement, and time-to-expiry to ensure high-convict
# Stats: {"live_returns": [-1.0, -1.0, -1.0, 1.890848, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.822350025177002, -1.0, -0.7043820023536682, -1.0, 2.39541, -1.0, -1.0, -0.8161702752113342, -1.0, -0.853145956993103, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7479773163795471, 2.39541, -1.0, 1.4562249999999999, -1.0, -1.0, -0.910955011844635, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -32.54873085021973, "live_wins": 8, "live_total": 61, "age": 82, "rounds_underwater": 56, "abstain_rounds": 0, "backtest_score": -0.19178850304610412, "backtest_acc": 0.5211498973305955, "backtest_n": 1802}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    velocity, exchange agreement, and the congruence between flow and book.
    """
    # Work on a copy to avoid modifying the original dataframe
    df_work = df.copy()

    # 1. Calculate velocities (diffs) within each market window
    # These capture the immediate momentum of price and aggressive flow
    df_work['delta_velocity'] = df_work.groupby('market_slug')['sum_delta'].diff().fillna(0)
    df_work['price_velocity'] = df_work.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. OBI Agreement: A measure of consensus across the three exchanges
    # Range: [-3, 3]. 3 = all exchanges have positive OBI, -3 = all negative.
    df_work['obi_agreement'] = (
        np.sign(df_work['binance_obi']) + 
        np.sign(df_work['coinbase_obi']) + 
        np.sign(df_work['kraken_obi'])
    ).astype(float)

    # 3. Delta-OBI Congruence: Measures if aggressive flow is being 
    # supported (same sign) or absorbed (opposite sign) by the passive book.
    df_work['delta_obi_congruence'] = df_work['sum_delta'] * df_work['mean_obi']

    # Drop metadata but keep pm_costs and seconds_remaining for model/strategy context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df_work.drop(columns=[c for c in cols_to_drop if c in df_work.columns])

    # Final cleanup: handle NaNs and infinities resulting from diffs or divisions
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 45.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Volatility-Adaptive Consensus Gating (VACG).
    The strategy uses a dynamic hurdle that increases during periods of 
    high volatility, exchange disagreement, or low time-to-expiry.
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

    # 2. Construct a Dynamic Hurdle
    # The base hurdle is 3% edge. We increase this requirement based on risk factors.
    hurdle = 0.03

    # Factor A: Exchange Disagreement (OBI Spread)
    # If exchanges disagree on direction, the market is fragmented and risky.
    if features_row.get('obi_spread', 0) > 0.25:
        hurdle += 0.02

    # Factor B: Volatility Regime
    # In high volatility, we require a larger margin of error.
    if features_row.get('mean_volatility', 0) > 0.0015:
        hurdle += 0.02

    # Factor C: Consensus Strength
    # If the OBI agreement is weak (near zero), we increase the hurdle.
    agreement = features_row.get('obi_agreement', 0)
    if abs(agreement) < 1.0:
        hurdle += 0.02

    # Factor D: Time Decay
    # As the window closes, price noise increases relative to the 5-min outcome.
    if seconds_remaining < 60:
        hurdle += 0.03

    # 3. Execution Logic
    if best_ev > hurdle:
        # Scale bet size based on how much the edge exceeds the dynamic hurdle.
        # We use a conservative multiplier (0.7) to manage variance.
        bet_fraction = (best_ev / hurdle) * 0.7
        
        # Ensure bet_fraction is within [0.1, 1.0]
        bet_fraction = min(1.0, max(0.1, bet_fraction))
        
        return side, bet_fraction

    return None, 0.0