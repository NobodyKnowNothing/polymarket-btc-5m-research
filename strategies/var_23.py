# Var 23 | BT: -0.793 | Live PnL: $-0.98 | Trades: 61
# Desc: Implements "Coordinated Impulse Capture (CIC)," which uses cross-exchange price dispersion and delta-OBI alignment to dynamically scale entry hurdles and bet sizes, penalizing trades occurring during
# Stats: {"live_returns": [-0.14846345782279968, 2.0514159202575684, -0.2062726765871048, 1.0480159521102905, -0.1530638486146927, -0.4777463376522064, -0.28053247928619385, -0.34018394351005554, -0.12242148816585541, 0.48572707176208496, -0.4522601366043091, 0.9568076133728027, -0.16860966384410858, -0.26483699679374695, -0.13891085982322693, -0.15103323757648468, -0.2496354579925537, 0.4469940960407257, -0.522689163684845, -0.3510911166667938, -0.44360798597335815, -0.72552889585495, -0.5247260332107544, -1.0, 0.6715009808540344, -0.5283848643302917, -0.25100448727607727, -0.18810197710990906, 0.470556378364563, 0.7107056975364685, -0.15411415696144104, 0.30847814679145813, -0.3077378571033478, -0.3173331022262573, -0.12586599588394165, -0.29323267936706543, -0.23704098165035248, -0.19334188103675842, -0.18834009766578674, 0.45087310671806335, -0.5992169976234436, -0.13610081374645233, -0.1380363404750824, 0.01393115520477295, 0.19864967465400696, 0.6996333599090576, 0.2628058195114136, 0.2551954686641693, -0.31392502784729004, -0.20939955115318298, 0.033259063959121704, 1.185654878616333, -0.3663020133972168, -0.11193610727787018, -0.1008366048336029, -0.18114697933197021, 1.33916437625885, -0.15237507224082947, -0.34687483310699463, -0.1654224991798401, -0.24314308166503906], "live_pnl": -0.9814593195915222, "live_wins": 18, "live_total": 61, "age": 133, "rounds_underwater": 40, "abstain_rounds": 0, "backtest_score": -0.7928878646969238, "backtest_acc": 0.6001803020058598, "backtest_n": 1957}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, adding 
    cohesion and congruence metrics to capture market coordination.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Flow Congruence: Alignment between aggressive flow (delta) and book imbalance (OBI)
    # Positive if they agree, negative if they conflict.
    X['flow_congruence'] = X['mean_delta'] * X['mean_obi']

    # 2. Volatility-Adjusted Flow: Measures the intensity of delta relative to the current regime
    X['vol_delta_ratio'] = X['mean_delta'] / (X['mean_volatility'] + 1e-7)

    # 3. OBI Magnitude: The absolute strength of the order book imbalance
    X['obi_strength'] = X['mean_obi'].abs()

    # 4. Price Dispersion: Relative spread between exchanges (as a proxy for fragmentation)
    # Using price_std directly is more robust across different BTC price levels
    X['rel_price_dispersion'] = X['price_std'] / (X['binance_price'] + 1e-7)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Coordinated Impulse Capture (CIC).
    The hurdle increases when price dispersion is high (fragmentation) 
    or when delta and OBI are misaligned (low congruence).
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

    # 2. Extract engineered coordination metrics
    # cohesion: 1.0 is perfect price agreement, lower is fragmented
    # We use price_std directly to avoid scaling issues with rel_dispersion
    price_std = features_row.get('price_std', 0.0)
    cohesion = 1.0 / (1.0 + price_std)
    
    # congruence: 1.0 is perfect delta/OBI alignment, 0.0 is no alignment
    congruence = abs(features_row.get('flow_congruence', 0.0))

    # 3. Calculate Dynamic Hurdle
    # Base hurdle is 2.5%.
    # Penalty for fragmentation (high price_std) up to 5%.
    # Penalty for flow misalignment (low congruence) up to 4%.
    fragmentation_penalty = 0.05 * (1.0 - cohesion)
    misalignment_penalty = 0.04 * (1.0 - congruence)
    
    dynamic_hurdle = 0.025 + fragmentation_penalty + misalignment_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: reduce size as we approach the 5-min resolution
        time_factor = max(0.5, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale based on how much the edge exceeds the hurdle,
        # but also heavily weight the size by the 'cohesion' of the market.
        # A fragmented market (low cohesion) gets much smaller bets.
        bet_fraction = (best_ev / dynamic_hurdle) * cohesion * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0