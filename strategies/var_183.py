# Var 183 | BT: 0.385 | Live PnL: $-10.09 | Trades: 60
# Desc: Implements "Elasticity-Gated Absorption (EGA)," which uses the relationship between aggressive flow (Delta) and price movement to distinguish between sustainable trends (high elasticity) and high-conv
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.04551428556442261, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.06987699999999997, -1.0, -1.0, 0.05925399999999992, 0.05925399999999992, -1.0, 1.6554160000000002, -1.0, -1.0, 0.12663000000000002, -1.0, -1.0, 0.06987699999999997, -1.0, 3.1194439999999997, -1.0, 1.3964720000000002, 0.15118699999999996, 0.028801262378692627, 0.10246717929840088, 12.389175, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7995867729187012, 0.045856773853302, -1.0, -1.0, -1.0, 1.5854279999999998, 4.245841, -1.0, 1.7294070000000001, -1.0, -0.7079088687896729, 2.39541, -1.0, -1.0, 1.1375739999999999, -1.0, -1.0, -1.0], "live_pnl": -10.094609260559082, "live_wins": 19, "live_total": 60, "age": 74, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": 0.3846787873775631, "backtest_acc": 0.5460657823005634, "backtest_n": 4368}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between aggressive flow and price elasticity.
    """
    X = df.copy()

    # 1. Price Change (within the market window)
    # We use groupby to ensure we don't calculate diffs across different markets
    X['binance_price_change'] = X.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Alignment: Sign of (Delta * Price Change)
    # 1.0 = Trend (Delta and Price move together), -1.0 = Absorption (Delta and Price diverge)
    X['alignment'] = np.sign(X['mean_delta']) * np.sign(X['binance_price_change'])

    # 3. Absorption Ratio: Magnitude of Delta relative to Price Change
    # High ratio = high absorption (price is resisting heavy flow)
    # We clip the denominator to prevent division by zero and extreme outliers
    X['absorption'] = (X['mean_delta'].abs() / X['binance_price_change'].abs().clip(lower=1e-4))
    X['absorption'] = X['absorption'].clip(upper=1000)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for capturing non-linear microstructure interactions."""
    return {
        'n_estimators': 1200,
        'max_depth': 3,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using "Elasticity-Gated Absorption (EGA)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Identify the regime: 
       - Trend (alignment > 0): Follow the model's direction.
       - Absorption (alignment < 0): The model's direction is likely a reversal 
         of the current aggressive flow. We boost conviction here.
    3. Dynamic Sizing:
       - Scale by EV magnitude.
       - Scale by absorption strength if in a reversal regime.
       - Scale down as time runs out to avoid late-window volatility.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Base hurdle for entry
    hurdle = 0.035
    
    side = None
    best_ev = 0.0
    
    # Extract engineered features
    alignment = features_row.get('alignment', 0.0)
    absorption = features_row.get('absorption', 0.0)

    # Determine best side and base conviction
    if ev_up > ev_down and ev_up > hurdle:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up and ev_down > hurdle:
        side = 'Down'
        best_ev = ev_down
    
    if side:
        # Conviction logic:
        # If alignment < 0, it means the price is moving AGAINST the delta (Absorption).
        # This is a high-conviction signal for a reversal.
        conviction = 1.0
        if alignment < 0:
            # Boost conviction based on the magnitude of absorption
            # We use a logarithmic scale to prevent extreme absorption from over-leveraging
            conviction = 1.0 + min(0.5, math.log1p(absorption) * 0.1)
        
        # 1. Edge Scaling: target 5% EV for full bet
        edge_scale = best_ev / 0.05
        
        # 2. Time Decay: Reduce size significantly in the last 90 seconds
        time_factor = 1.0 if seconds_remaining > 90 else (seconds_remaining / 90.0)
        
        bet_fraction = edge_scale * conviction * time_factor
        
        # Final bounds check
        return side, min(1.0, max(0.01, bet_fraction))

    return None, 0.0