# Var 24 | BT: 2.324 | Live PnL: $-1.07 | Trades: 11
# Desc: Implements "Microstructure Cohesion-Gated Value Capture (MCGVC)," which requires high cross-exchange OBI magnitude agreement and scales the entry hurdle based on both volatility and the lack of exchan

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange cohesion and microstructure pressure.
    """
    # Work on a copy to avoid side effects
    X = df.copy()

    # 1. Convert categorical volatility to numeric for the model
    # We use a small base vol for 'low' and a higher one for 'high'
    X['vol_numeric'] = X['mean_volatility'].apply(lambda x: 0.002 if x == 'low' else 0.01)

    # 2. Microstructure Cohesion: Do the exchanges agree on the magnitude of imbalance?
    # We use the inverse of the OBI spread as a proxy for cohesion.
    # A spread of 0 means perfect agreement; a spread of 1 means maximum disagreement.
    X['obi_cohesion'] = 1.0 / (1.0 + X['obi_spread'])

    # 3. Delta-OBI Pressure: The product of net aggressive flow and the book imbalance.
    # This captures whether aggressive buying is happening into a bid-heavy book.
    X['delta_obi_pressure'] = X['sum_delta'] * X['mean_obi']

    # 4. Temporal Velocity: Capture changes in momentum using groupby to avoid leakage
    group = X.groupby('market_slug')
    X['delta_velocity'] = group['sum_delta'].diff().fillna(0)
    X['price_velocity'] = group['binance_price'].diff().fillna(0)
    X['obi_velocity'] = group['mean_obi'].diff().fillna(0)

    # 5. Price-to-Volatility Ratio: Is the current price movement significant?
    X['price_vol_ratio'] = X['price_velocity'] / (X['vol_numeric'] + 1e-7)

    # Drop metadata columns to return only features
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup: Replace NaNs/Infs with 0
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Microstructure Cohesion-Gated Value Capture.
    
    The hurdle for entry is dynamically adjusted by:
    1. Volatility (higher volatility = higher hurdle)
    2. Cohesion (lower exchange agreement = higher hurdle)
    
    This ensures we only bet when the model's edge is large enough to overcome
    both market noise and lack of consensus.
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

    # 2. Retrieve engineered context
    vol = features_row.get('vol_numeric', 0.005)
    cohesion = features_row.get('obi_cohesion', 0.5)

    # 3. Calculate Dynamic Risk-Adjusted Hurdle
    # Base hurdle is 2%. 
    # Volatility penalty: adds up to 4% to the hurdle.
    # Cohesion penalty: adds up to 6% to the hurdle if exchanges strongly disagree.
    base_hurdle = 0.02
    vol_penalty = vol * 4.0
    cohesion_penalty = 0.06 * (1.0 - cohesion)
    
    total_hurdle = base_hurdle + vol_penalty + cohesion_penalty

    # 4. Execution Logic
    if best_ev > total_hurdle:
        # Use quadratic scaling for conviction, but dampen it slightly by cohesion
        # to favor trades where all exchanges are in sync.
        conviction = (best_ev / total_hurdle) ** 2
        bet_fraction = min(1.0, conviction * cohesion)
        
        # Ensure we don't return a tiny fraction that isn't worth the effort
        if bet_fraction < 0.1:
            return None, 0.0
            
        return side, bet_fraction

    return None, 0.0