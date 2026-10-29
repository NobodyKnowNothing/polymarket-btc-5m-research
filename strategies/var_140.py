# Var 140 | BT: 0.600 | Live PnL: $-34.79 | Trades: 60
# Desc: Implements "Volatility-Compressed Dislocation Breakout (VCDB)," which targets high-conviction entries by identifying when the price-to-Polymarket dislocation begins to expand rapidly during periods of
# Stats: {"live_returns": [-1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, 1.049452, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, 1.1846420000000002, -1.0, 0.12663000000000002, 1.049452, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.1846420000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.950496, -1.0, -1.0, -1.0], "live_pnl": -34.79271, "live_wins": 10, "live_total": 60, "age": 113, "rounds_underwater": 57, "abstain_rounds": 1, "backtest_score": 0.5995633765935664, "backtest_acc": 0.5903781429419916, "backtest_n": 3273}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the velocity of price-implied dislocation relative to volatility.
    """
    df = df.copy()

    # 1. Calculate Synthetic Mid-Price
    df['synth_price'] = (df['binance_price'] * 0.5) + (df['coinbase_price'] * 0.3) + (df['kraken_price'] * 0.2)

    # 2. Price-Implied Return (Proxy for fair probability)
    df['price_implied_return'] = (df['synth_price'] - df['strike_price']) / df['strike_price']

    # 3. Dislocation: The gap between price-implied return and PM cost-implied return
    # If dislocation > 0, price is higher than PM cost implies (Buy Up)
    # If dislocation < 0, price is lower than PM cost implies (Buy Down)
    df['dislocation'] = df['price_implied_return'] - (df['pm_cost_up'] - 0.5)

    # 4. Dislocation Velocity (Rate of change of the dislocation)
    df['dislocation_velocity'] = df.groupby('market_slug')['dislocation'].diff().fillna(0)

    # 5. Volatility Compression Factor
    # High value when mean_volatility is low.
    df['vol_compression'] = 1.0 / (df['mean_volatility'] + 1e-6)

    # 6. Breakout Score: Captures rapid dislocation expansion in low-vol regimes
    df['breakout_score'] = df['dislocation_velocity'] * df['vol_compression']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Compressed Dislocation Breakout (VCDB)".

    Logic:
    - Calculate raw EV for both sides.
    - Identify the side with the best EV.
    - Alignment Check: A trade is amplified if the dislocation velocity is moving 
      in the same direction as the predicted side (e.g., betting 'Up' while 
      dislocation is increasing).
    - Volatility-Gated Entry: Uses a dynamic threshold that increases as time 
      runs out, requiring higher conviction in the final seconds.
    - Dynamic Sizing: Scales bet fraction based on the amplified effective EV.
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
    velocity = features_row.get('dislocation_velocity', 0.0)

    # 1. Alignment Multiplier
    # If we are betting 'Up', we want dislocation velocity to be positive.
    # If we are betting 'Down', we want dislocation velocity to be negative.
    if (side == 'Up' and velocity > 0) or (side == 'Down' and velocity < 0):
        alignment_multiplier = 1.8
    else:
        alignment_multiplier = 0.4  # Penalize if momentum contradicts the model

    # 2. Calculate Effective EV
    effective_ev = best_ev * alignment_multiplier

    # 3. Dynamic Thresholding (Increases as time runs out)
    # At 300s, threshold is ~0.03. At 0s, threshold is ~0.08.
    time_progress = seconds_remaining / 300.0
    base_threshold = 0.03
    dynamic_threshold = base_threshold + (1.0 - time_progress) * 0.05

    if effective_ev > dynamic_threshold:
        # 4. Dynamic Bet Sizing
        # Scale bet fraction based on effective EV. 
        # An effective EV of 0.05 results in a 1.0 bet.
        bet_fraction = effective_ev / 0.05
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0