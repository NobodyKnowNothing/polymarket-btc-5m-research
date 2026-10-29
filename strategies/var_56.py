# Var 56 | BT: -0.385 | Live PnL: $-17.05 | Trades: 61
# Desc: Dispersion-Normalized Edge Gating (DNEG): Scales the required edge hurdle based on relative cross-exchange price dispersion to account for price uncertainty and signal noise.
# Stats: {"live_returns": [0.8327068090438843, -1.0, -0.5279491543769836, -0.8087596893310547, -1.0, -0.48178449273109436, -0.7079717516899109, 2.39541, 0.06305402517318726, -0.7308365106582642, -1.0, -0.5455971956253052, -0.7786691188812256, 1.5191249999999998, -0.5201607942581177, 2.39541, -0.6168490648269653, -0.9531511068344116, 1.4562249999999999, -1.0, 0.8835580348968506, -1.0, -1.0, 2.39541, -1.0, -0.5989528894424438, -1.0, -0.6821906566619873, 0.071175217628479, -1.0, -1.0, 3.1194439999999997, -0.5983208417892456, -1.0, -1.0, -1.0, -1.0, -0.7161376476287842, -0.4697766900062561, -0.7869038581848145, -1.0, 2.39541, 0.6527571678161621, -0.9079484939575195, -1.0, 0.6538286209106445, 1.4562249999999999, -1.0, -1.0, -0.6032024621963501, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -1.0, -0.6384055018424988, -1.0, -1.0, -0.6489945650100708, -1.0], "live_pnl": -17.05368423461914, "live_wins": 15, "live_total": 61, "age": 103, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": -0.3854651237367155, "backtest_acc": 0.561700656757691, "backtest_n": 1984}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    relative dispersion and flow-to-volatility ratios.
    """
    df = df.copy()

    # 1. Relative Price Dispersion: How much do exchanges disagree relative to price?
    # High dispersion = high uncertainty in the 'true' price.
    df['rel_price_dispersion'] = df['price_std'] / (df['binance_price'] + 1e-6)

    # 2. Flow-to-Volatility Ratio: Is the aggressive flow significant relative to current volatility?
    df['flow_vol_ratio'] = df['sum_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. OBI-Delta Confluence: Alignment between book pressure and aggressive flow.
    df['obi_delta_confluence'] = df['mean_obi'] * df['sum_delta']

    # 4. Momentum-Flow Divergence: Is price moving without aggressive flow?
    # (High momentum with low delta suggests a possible liquidity gap or exhaustion)
    df['mom_delta_divergence'] = df['mean_momentum'] / (abs(df['sum_delta']) + 1e-6)

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == float:
            # Cap extreme outliers to prevent XGBoost instability
            X.loc[X[col] > 1e10, col] = 0
            X.loc[X[col] < -1e-10, col] = 0
            
    # Replace any remaining infs from division
    X = X.replace([np.inf, -np.inf], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Dispersion-Normalized Edge Gating (DNEG)".
    The required edge (hurdle) is dynamically increased when exchange price 
    dispersion is high, as high dispersion indicates a high-noise regime.
    """
    # 1. Calculate raw Expected Value (EV)
    eps = 1e-6
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Dynamic Hurdle Calculation
    # We use the relative price dispersion as a proxy for market uncertainty.
    # As dispersion increases, we demand a much higher margin of safety.
    rel_dispersion = features_row.get('rel_price_dispersion', 0.0)
    
    # Base hurdle of 2% + a penalty scaled by dispersion.
    # If dispersion is 0.01% (0.0001), hurdle is 0.02 + (0.0001 * 500) = 0.07 (7%)
    hurdle = 0.02 + (rel_dispersion * 500.0)

    # 3. Execution Decision
    if best_ev < hurdle:
        return None, 0.0

    # 4. Dynamic Sizing
    # Scale bet size by:
    # - The ratio of actual edge to required hurdle
    # - Time decay: reduce size as the window closes to avoid last-second volatility spikes
    time_factor = max(0.0, seconds_remaining / 300.0)
    edge_strength = best_ev / hurdle
    
    bet_fraction = edge_strength * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to ensure we aren't trading tiny, insignificant edges
    if bet_fraction < 0.10:
        return None, 0.0

    return side, bet_fraction