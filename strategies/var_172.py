# Var 172 | BT: 0.112 | Live PnL: $-13.28 | Trades: 60
# Desc: Implements "Asymmetric Microstructure-Funding Hurdle (AMFH)," which uses a spread-to-volatility noise index and funding-rate-induced asymmetry to dynamically adjust the required EV hurdle for Up vs Do
# Stats: {"live_returns": [-1.0, 2.518758, -1.0, -1.0, -1.0, 2.518758, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -0.9844963550567627, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8880487680435181, -0.9164069294929504, -1.0, -1.0, -0.8956472873687744, -1.0, -1.0, 3.1194439999999997, -0.921744167804718, -1.0, -0.8984217643737793, -0.8732647895812988, -1.0, -1.0, -0.9990612864494324, -1.0, 4.245841, -1.0, 1.3454298973083496, -1.0, 2.39541, -1.0, -1.0, -1.0, 2.280573, 2.518758, 2.39541, -1.0, -1.0, -1.0, 2.950496, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0], "live_pnl": -13.283166885375977, "live_wins": 13, "live_total": 60, "age": 173, "rounds_underwater": 50, "abstain_rounds": 0, "backtest_score": 0.11174990998213558, "backtest_acc": 0.5820456217807212, "backtest_n": 2293}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between microstructure noise and funding-driven asymmetry.
    """
    df = df.copy()

    # 1. Spread-to-Volatility Noise: 
    # Measures how much the bid/ask and price spreads dominate the actual price movement.
    # High values indicate a "noisy" or "illiquid" environment where signals are less reliable.
    df['spread_noise'] = (df['price_spread'] + df['obi_spread']) / (df['price_std'] + 1e-6)

    # 2. Funding Pressure:
    # Direct capture of the perp market's bias.
    df['funding_pressure'] = df['mean_funding']

    # 3. Delta-Price Efficiency:
    # How much aggressive flow (delta) is actually moving the price.
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters for robust signal extraction."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Asymmetric Microstructure-Funding Hurdle (AMFH)".

    Logic:
    1. Calculate Total Noise: Combine the spread-to-volatility ratio with a time-decay 
       factor (noise increases as the market window closes).
    2. Asymmetric Hurdle: 
       - Base hurdle is adjusted upward by the noise index.
       - If funding is positive (longs paying), the 'Up' hurdle is significantly 
         increased (harder to bet Up) and the 'Down' hurdle is lowered (easier to bet Down)
         to capture potential "long squeezes".
    3. Dynamic Sizing: Scale the bet fraction by the magnitude of the edge and 
       inversely by the total noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    noise_index = features_row.get('spread_noise', 0.0)
    funding = features_row.get('funding_pressure', 0.0)
    
    # 1. Calculate Total Noise (Microstructure + Temporal)
    # As seconds_remaining -> 0, time_noise increases.
    time_noise = 1.0 / (seconds_remaining + 1.0)
    total_noise = noise_index + time_noise

    # 2. Determine Asymmetric Hurdles
    # base_hurdle is the minimum EV required in a perfect, low-noise, neutral-funding market.
    base_hurdle = 0.03
    
    # The funding multiplier (500) converts small funding rates (e.g., 0.0001) 
    # into meaningful hurdle shifts (e.g., 0.05).
    funding_adj = funding * 500.0
    
    # If funding is positive, hurdle_up increases, making it harder to bet Up.
    # If funding is negative, hurdle_down increases, making it harder to bet Down.
    hurdle_up = base_hurdle * (1.0 + total_noise) + max(0.0, funding_adj)
    hurdle_down = base_hurdle * (1.0 + total_noise) + max(0.0, -funding_adj)

    side = None
    best_ev = 0.0

    # 3. Decision Logic
    if ev_up > hurdle_up:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > hurdle_down:
        side = 'Down'
        best_ev = ev_down

    # 4. Execution & Sizing
    if side:
        # Scale bet by:
        # a) Edge magnitude (target 4% EV for full size)
        # b) Inverse of noise (reduce size in volatile/spread-heavy regimes)
        bet_fraction = (best_ev / 0.04) * (1.0 / (1.0 + total_noise))
        
        # Clamp bet between 5% and 100%
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0