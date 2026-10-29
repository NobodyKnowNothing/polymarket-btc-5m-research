# Var 129 | BT: 2.938 | Live PnL: $0.00 | Trades: 0
# Desc: Implements "Tri-Factor Flow-Book-Price Coherence (TFBC)," which filters for trades where aggressive order flow, order book imbalance, and price movement are all in directional alignment, while penaliz
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 59, "rounds_underwater": 0, "abstain_rounds": 59, "backtest_score": 2.9381127119141004, "backtest_acc": 0.5627430910951894, "backtest_n": 178}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the coherence between order flow, book state, and price action.
    """
    df = df.copy()

    # 1. Signed Price Difference: Directional distance from strike
    df['signed_price_diff'] = df['binance_price'] - df['strike_price']

    # 2. Flow-Book Congruence: Do aggressive trades align with the book imbalance?
    # Range: [-1, 1]. 1 = delta and OBI agree, -1 = they disagree, 0 = neutral.
    df['flow_book_congruence'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 3. Flow-Price Congruence: Does the aggressive flow align with the price move?
    # Range: [-1, 1]. 1 = delta and price_diff agree, -1 = they disagree.
    df['flow_price_congruence'] = np.sign(df['mean_delta']) * np.sign(df['signed_price_diff'])

    # 4. Relative Dispersion: Price spread normalized by volatility.
    # High value = high noise/inefficiency; Low value = stable/efficient.
    df['relative_dispersion'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 50.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Factor Flow-Book-Price Coherence (TFBC)".

    Logic:
    - Calculate raw EV for both sides.
    - Coherence Score: A value from 0.0 to 1.0 based on the alignment of 
      Delta, OBI, and Price Direction.
    - Stability Factor: Penalizes trades when exchange price dispersion 
      is high relative to the current volatility regime.
    - Time Factor: Gradually reduces conviction as the market window closes.
    - Effective EV: The product of raw EV, coherence, stability, and time.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 1. Coherence Score (0.0 to 1.0)
    # We use a mathematical mapping: 
    # Both agree: (1 + 1 + 2)/4 = 1.0
    # One agrees, one disagrees: (-1 + 1 + 2)/4 = 0.5
    # Both disagree: (-1 - 1 + 2)/4 = 0.0
    fbc = features_row.get('flow_book_congruence', 0.0)
    fpc = features_row.get('flow_price_congruence', 0.0)
    coherence_score = (fbc + fpc + 2.0) / 4.0

    # 2. Stability Factor (0.0 to 1.0)
    # High dispersion relative to vol reduces the stability factor.
    rel_dispersion = features_row.get('relative_dispersion', 0.0)
    stability_factor = 1.0 / (1.0 + rel_dispersion)

    # 3. Time Factor (0.0 to 1.0)
    # Reduces bet size as the window approaches zero to avoid last-second whipsaws.
    time_factor = min(1.0, seconds_remaining / 300.0)

    # 4. Calculate Effective EV
    effective_ev = best_ev * coherence_score * stability_factor * time_factor

    # 5. Execution
    # We require a lower effective threshold because the multipliers are aggressive.
    threshold = 0.015
    if effective_ev > threshold:
        # Scale bet size: target 4% effective edge for a full bet.
        bet_fraction = (effective_ev / 0.04)
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0