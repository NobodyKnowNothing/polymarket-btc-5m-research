# Var 34 | BT: -0.280 | Live PnL: $-6.22 | Trades: 61
# Desc: Liquidity-Validated Momentum (LVM) — Filters momentum trades by calculating the price impact of aggressive flow; it scales down bets and increases entry hurdles during high-impact (low-liquidity) even
# Stats: {"live_returns": [0.03049302101135254, -1.0, 0.09182599999999996, 2.073133, -1.0, 0.13876599999999994, 1.890848, -0.9992635250091553, -1.0, 0.11476700000000006, -1.0, 0.038675000000000015, 0.08073200000000003, 1.6554160000000002, 3.727685, 4.245841, 0.10316799999999993, -1.0, -1.0, -1.0, 0.11476700000000006, -1.0, 1.28551, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0081479999999998, 0.0844164490699768, 0.038675000000000015, 0.09984391927719116, -1.0, 1.059657096862793, 3.1194439999999997, -1.0, -1.0, -1.0, -0.7969598770141602, -0.638218879699707, -1.0, 2.39541, 1.28551, -1.0, -1.0, 1.092555, 1.4562249999999999, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0], "live_pnl": -6.223790168762207, "live_wins": 25, "live_total": 61, "age": 75, "rounds_underwater": 20, "abstain_rounds": 0, "backtest_score": -0.27954330977835645, "backtest_acc": 0.517387944358578, "backtest_n": 1926}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically
    calculating the 'Price Impact' to distinguish between sustainable 
    trends and low-liquidity spikes.
    """
    df = df.copy()

    # 1. Calculate Price Velocity per exchange
    # Grouping by market_slug ensures diffs are calculated within the same 5-min window
    df['binance_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['coinbase_vel'] = df.groupby('market_slug')['coinbase_price'].diff().fillna(0)
    df['kraken_vel'] = df.groupby('market_slug')['kraken_price'].diff().fillna(0)

    # 2. Price Impact Calculation
    # High Impact = Large Delta (aggressive flow) moving the price very little (low liquidity).
    # Low Impact = Large Delta moving the price significantly (high liquidity/efficient trend).
    # We use log1p to compress the scale for the XGBoost model.
    total_vel = abs(df['binance_vel']) + abs(df['coinbase_vel']) + abs(df['kraken_vel'])
    df['price_impact'] = np.log1p(abs(df['mean_delta']) / (total_vel + 1e-6))

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Validated Momentum".
    The strategy scales the entry hurdle and bet size based on the 'price_impact'.
    In high-impact (low-liquidity) environments, the hurdle is higher and 
    the bet size is smaller to avoid being caught in liquidity-driven fakeouts.
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

    # 2. Extract Liquidity Metric
    # price_impact was log-transformed in engineer_features
    impact = features_row.get('price_impact', 0.0)
    
    # liquidity_factor: 1.0 = high liquidity, approaches 0.0 = very low liquidity
    liquidity_factor = 1.0 / (1.0 + impact)

    # 3. Dynamic Hurdle
    # Base hurdle is 2.0%, but it increases as liquidity decreases.
    # High impact (low liquidity) -> higher hurdle.
    hurdle = 0.02 / liquidity_factor

    if best_ev < hurdle:
        return None, 0.0

    # 4. Execution & Sizing
    # We scale the bet size by the liquidity factor and the time remaining.
    # We want to be most aggressive when the edge is high AND liquidity is high.
    
    # Time-decay component
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # Final bet fraction: 
    # (Edge/Hurdle) provides the base scaling, 
    # liquidity_factor ensures we don't over-leverage in thin markets,
    # time_factor reduces exposure as the window closes.
    bet_fraction = (best_ev / hurdle) * liquidity_factor * time_factor
    
    # Cap bet fraction at 1.0
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction