# Var 32 | BT: -0.116 | Live PnL: $-0.11 | Trades: 64
# Desc: Implements "Binance-Led Dominance Capture (BLDC)," which shifts focus from price velocity to order-flow dominance, filtering for trades where Binance's aggressive delta is the primary driver of market
# Stats: {"live_returns": [0.21097947657108307, 0.32690170407295227, -0.10665977001190186, 0.13494624197483063, -0.18376100063323975, -0.10894817858934402, -0.271176278591156, -0.05321846902370453, 1.0773690938949585, -0.12662656605243683, -0.18184225261211395, 0.09343783557415009, -0.1591554433107376, 0.009224683046340942, -0.3785724639892578, 0.4278690814971924, -0.11370642483234406, 0.04032149910926819, -0.14652441442012787, 0.16145142912864685, 0.9050981402397156, 0.9395754933357239, -0.08017327636480331, -0.15763263404369354, -0.16011421382427216, 0.007389239966869354, -0.1732698678970337, -0.11114036291837692, 0.014491885900497437, -0.1162436380982399, -0.2478853464126587, -0.18522866070270538, -0.0854656994342804, -0.14253810048103333, 1.0056596994400024, -0.06734644621610641, -0.1950358748435974, 0.15621700882911682, -0.39021700620651245, 0.45409223437309265, 0.017556577920913696, 0.007049232721328735, 0.10395991057157516, -0.13541346788406372, -0.07629984617233276, -0.06488421559333801, 0.002730179578065872, -0.09112787991762161, -0.08419226109981537, -0.13484659790992737, -0.2929704487323761, -0.09445355087518692, 0.007383815944194794, -0.1017719954252243, -0.0758390799164772, 0.10204006731510162, -0.1297113597393036, -0.09923353046178818, -0.11928693950176239, -0.14751358330249786, -0.0849473625421524, -0.41728606820106506, -0.09552426636219025, -0.1279730349779129], "live_pnl": -0.11001282930374146, "live_wins": 22, "live_total": 64, "age": 133, "rounds_underwater": 1, "abstain_rounds": 5, "backtest_score": -0.11580241074929, "backtest_acc": 0.572659616137877, "backtest_n": 1569}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Binance's role as the primary liquidity driver and the 'conflict' 
    between exchange flows.
    """
    df = df.copy()

    # 1. Binance Flow Dominance: How much of the total market delta is coming from Binance?
    # Positive if Binance is leading the aggressive flow.
    df['binance_delta_rel'] = df['binance_delta'] / (df['mean_delta'].abs() + 1e-6)

    # 2. OBI Congruence: Do Binance's book pressures align with the broader market?
    # High positive = Binance and market are in sync; High negative = Binance is a contrarian.
    df['obi_agreement'] = df['binance_obi'] * df['mean_obi']

    # 3. Exchange Flow Conflict: Measures the divergence between Binance and the mean flow.
    # High values indicate a "war" between exchanges (e.g., Binance buying, others selling).
    df['flow_conflict'] = (df['binance_delta'] - df['mean_delta']).abs()

    # 4. Volatility-Adjusted Price Dispersion: Is the cross-exchange spread normal for this vol?
    df['spread_to_vol_ratio'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 5. Delta-Price Efficiency: How much price moves per unit of delta.
    # This helps the model identify 'absorption' (low efficiency) vs 'breakouts' (high efficiency).
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Binance-Led Dominance Capture".
    The strategy seeks 'clean' breakouts where Binance delta leads the market 
    without significant friction or conflict from other exchanges.
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
    binance_delta_rel = features_row.get('binance_delta_rel', 0.0)
    obi_agreement = features_row.get('obi_agreement', 0.0)
    flow_conflict = features_row.get('flow_conflict', 0.0)
    spread_to_vol = features_row.get('spread_to_vol_ratio', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)

    # 3. Dynamic Hurdle
    # Higher hurdle in high-conflict or high-spread environments
    base_hurdle = 0.025
    conflict_penalty = min(0.04, flow_conflict / (abs(mean_delta) + 1e-6) * 0.01)
    hurdle = base_hurdle + conflict_penalty

    if best_ev < hurdle:
        return None, 0.0

    # 4. Multiplier Logic (The "Cleanliness" Filter)
    multiplier = 1.0

    # A) Flow Dominance: Is Binance actually the driver?
    # If Binance delta is moving opposite to the market mean, it's a fight.
    if (side == 'Up' and binance_delta_rel < 0.5) or (side == 'Down' and binance_delta_rel > -0.5):
        multiplier *= 0.4

    # B) OBI Alignment: Is the book supporting the aggressive flow?
    # If delta is Up but OBI is heavily Down, we are hitting a wall (absorption).
    if (side == 'Up' and obi_agreement < 0) or (side == 'Down' and obi_agreement > 0):
        multiplier *= 0.3

    # C) Exchange Conflict: Are the other exchanges fighting this move?
    # If flow_conflict is high, the move is unstable.
    if flow_conflict > (abs(mean_delta) * 1.5):
        multiplier *= 0.5

    # D) Price Friction: Is the cross-exchange spread too wide?
    if spread_to_vol > 3.0:
        multiplier *= 0.5

    # 5. Execution & Sizing
    # Time-decay: Use a quadratic decay to aggressively reduce exposure in the final seconds.
    time_factor = (seconds_remaining / 300.0) ** 2
    
    # Final bet fraction: Scale by (Edge/Hurdle) and the cleanliness multiplier
    # We cap the bet_fraction to prevent over-leveraging on single outliers.
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction