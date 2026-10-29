# Var 173 | BT: 0.131 | Live PnL: $-21.39 | Trades: 60
# Desc: Implements "Microstructure Density-Adaptive Execution (MDAE)," which differentiates between "Vacuum" regimes (low order-book spread, high price impact) and "Wall" regimes (high order-book spread, high
# Stats: {"live_returns": [1.4562249999999999, -0.6572637557983398, -0.8084487318992615, -0.409771203994751, -0.389085054397583, 4.245841, -1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, 0.5441317558288574, -1.0, -1.0, -0.5062884092330933, -0.9598584771156311, -1.0, -1.0, 1.4562249999999999, -1.0, 0.7535826563835144, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.48564425110816956, 2.795066, -1.0, -1.0, -1.0, 1.5191249999999998, -0.8950511813163757, -1.0, -1.0, -1.0, -1.0, -0.6572637557983398, 0.012015581130981445, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.45932161808013916, -1.0, -1.0, -0.45932161808013916, -1.0, 1.6554160000000002, -0.45932161808013916, -1.0, -1.0], "live_pnl": -21.39092254638672, "live_wins": 12, "live_total": 60, "age": 91, "rounds_underwater": 34, "abstain_rounds": 0, "backtest_score": 0.1311409539098491, "backtest_acc": 0.5505968778696051, "backtest_n": 4414}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of delta (price impact) and the density of the order book.
    """
    df = df.copy()

    # 1. Price Change: The movement in the lead exchange (Binance)
    # We group by market_slug to ensure diffs don't leak across different markets.
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Delta Efficiency: How much price moves per unit of aggressive flow.
    # High efficiency = "Vacuum" (price slices through the book).
    # Low efficiency = "Absorption" (delta is being eaten by limit orders).
    df['delta_efficiency'] = df['price_change'] / (df['mean_delta'].abs() + 1e-6)

    # 3. Absorption Index: The ratio of aggressive flow to order book thickness.
    # High index = High flow hitting a thick book (High probability of reversal/absorption).
    # Low index = High flow hitting a thin book (High probability of breakout/vacuum).
    df['absorption_index'] = df['mean_delta'].abs() / (df['obi_spread'] + 1e-6)

    # 4. Momentum/Volatility Interaction: Normalizing momentum by the current regime.
    df['vol_normalized_momentum'] = df['mean_momentum'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle potential infs from division and NaNs from diffs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for capturing non-linear microstructure interactions."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.007,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Density-Adaptive Execution (MDAE)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Identify the Microstructure Regime via 'absorption_index':
       - VACUUM (Low absorption): Flow is slicing through a thin book. 
         High conviction in the model's directional prediction.
       - WALL (High absorption): Flow is being absorbed by a thick book. 
         Lower conviction; the price is likely to resist the trend.
    3. Dynamic Sizing:
       - Scale bet_fraction up in Vacuums (momentum is easier).
       - Scale bet_fraction down in Walls (momentum is harder).
       - Scale down as time runs out to avoid late-window noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine the best side
    if ev_up > ev_down and ev_up > 0.025:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up and ev_down > 0.025:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    absorption_idx = features_row.get('absorption_index', 1.0)
    obi_spread = features_row.get('obi_spread', 0.01)

    # 1. Regime Multiplier
    # High absorption_index or high obi_spread suggests a "Wall" (harder to move price).
    # Low absorption_index and low obi_spread suggests a "Vacuum" (easier to move price).
    if absorption_idx > 5.0 or obi_spread > 0.05:
        # --- WALL REGIME ---
        regime_multiplier = 0.6
    elif absorption_idx < 0.5 and obi_spread < 0.01:
        # --- VACUUM REGIME ---
        regime_multiplier = 1.4
    else:
        regime_multiplier = 1.0

    # 2. Time Decay Factor
    # As we approach the 5-minute mark, the probability of a sudden regime shift increases.
    time_factor = 1.0 if seconds_remaining > 45 else (seconds_remaining / 45.0)

    # 3. Final Bet Sizing
    # Base bet is scaled by (EV / Target_EV) * Regime * Time
    # We target 4% EV for a full bet.
    target_ev = 0.04
    bet_fraction = (best_ev / target_ev) * regime_multiplier * time_factor

    # Clamp bet_fraction between 0.05 and 1.0
    return side, min(1.0, max(0.05, bet_fraction))