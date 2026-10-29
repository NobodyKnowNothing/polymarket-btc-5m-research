# Var 38 | BT: -0.776 | Live PnL: $-13.98 | Trades: 61
# Desc: Regime-Adaptive Microstructure Validation (RAMV) — This strategy switches its validation logic based on the volatility regime: in low-volatility environments, it seeks "Efficient Momentum" (high price
# Stats: {"live_returns": [-1.0, 0.039315760135650635, -1.0, -1.0, -1.0, -0.6195141077041626, -0.8964579105377197, -0.9753183126449585, -0.5224089622497559, -0.749269962310791, -0.6029249429702759, 3.246485710144043, -0.4853147864341736, -0.3506302833557129, -1.0, -0.9972490072250366, 0.9629272222518921, -0.8288286328315735, 1.668431043624878, -0.2552787661552429, -1.0, 1.9791409999999998, -0.63525390625, -1.0, -1.0, -1.0, -0.39819565415382385, -1.0, -1.0, 0.03129652142524719, 0.3163521885871887, -1.0, 4.550646, 3.1194439999999997, -0.7506896257400513, -1.0, -0.21439048647880554, -1.0, -0.2932209372520447, -0.4502757489681244, -1.0, -1.0, 2.39541, 0.301968514919281, -0.27552106976509094, -0.5584597587585449, 0.7603151202201843, 0.959450364112854, -0.24247771501541138, -1.0, -1.0, -0.6258206963539124, -1.0, -1.0, 1.2686374187469482, -0.22620058059692383, -0.3825654089450836, -1.0, -0.621819794178009, -0.6211485862731934, -1.0], "live_pnl": -13.979413986206055, "live_wins": 14, "live_total": 61, "age": 83, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.7760006431721937, "backtest_acc": 0.5724033358605004, "backtest_n": 2024}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically 
    calculating 'delta_impact' (price movement efficiency) and 
    'obi_consensus' (agreement across exchanges).
    """
    df = df.copy()

    # 1. Price Velocity & Delta Impact
    # We calculate the price change on Binance specifically to measure efficiency
    df['binance_price_diff'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    
    # delta_impact: How much the price moves per unit of aggressive delta.
    # High impact = efficient trend/vacuum. Low impact = absorption/squeeze.
    df['delta_impact'] = df['binance_price_diff'] / (df['sum_delta'].abs() + 1e-6)

    # 2. OBI Consensus
    # Sum of signs of OBI across the three exchanges. 
    # Range: [-3, 3]. Higher magnitude = stronger cross-exchange agreement.
    df['obi_consensus'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    )

    # 3. Spread Velocity
    # Is the cross-exchange dispersion widening or narrowing?
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for capturing non-linear regime shifts."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 12.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Regime-Adaptive Microstructure Validation.
    
    Logic:
    - In Low Volatility: Validate the model's direction by requiring 'Efficient Momentum' 
      (High delta_impact and OBI consensus).
    - In High Volatility: Validate the model's direction by requiring 'Absorption' 
      (Low delta_impact despite high delta), targeting reversals.
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
    impact = features_row.get('delta_impact', 0.0)
    consensus = features_row.get('obi_consensus', 0.0)
    vol = features_row.get('mean_volatility', 0.001)
    delta = features_row.get('sum_delta', 0.0)

    # 3. Dynamic Hurdle
    # Higher hurdle in high vol to protect against noise
    hurdle = 0.025 + (0.04 * vol)
    if best_ev < hurdle:
        return None, 0.0

    # 4. Regime-Based Validation
    # We use a fixed threshold for volatility regime (0.001 as a proxy for 'high')
    # In a real production environment, this would be a rolling median.
    is_valid = False
    multiplier = 0.0

    if vol < 0.001:
        # LOW VOL REGIME: Look for "Clean Breakouts"
        # We want the price to move efficiently with the delta and have OBI agreement.
        if side == 'Up':
            is_valid = (impact > 0 and consensus > 0.5)
        else:
            is_valid = (impact < 0 and consensus < -0.5)
        
        multiplier = 1.0 if is_valid else 0.25
    else:
        # HIGH VOL REGIME: Look for "Absorption/Squeezes"
        # We want to see heavy flow (high delta) that is NOT moving the price (low impact).
        # This indicates a potential reversal or a massive liquidity trap.
        is_valid = (abs(impact) < 0.0005 and abs(delta) > 0)
        
        multiplier = 1.0 if is_valid else 0.3

    # 5. Execution & Sizing
    # Time-decay: Reduce bet size as the 5-minute window expires to avoid late-window noise.
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # Final bet fraction: Scale by (Edge/Hurdle) and the regime multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction