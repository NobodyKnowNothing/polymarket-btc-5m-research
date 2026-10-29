# Var 157 | BT: -0.778 | Live PnL: $-0.49 | Trades: 60
# Desc: Implements "Dynamic Risk-Adjusted Hurdle (DRAH)," which shifts the focus from signal detection to risk-mitigation by dynamically increasing the required Expected Value (EV) threshold during high-volat
# Stats: {"live_returns": [0.01349860429763794, 2.1024680137634277, -0.7026878595352173, -0.9881808757781982, 0.13446971774101257, 4.893558, -0.20240627229213715, -0.4334644079208374, -0.6042510867118835, -0.8257993459701538, -1.0, -0.4928940534591675, 1.5662868022918701, -0.43062666058540344, 0.024125173687934875, 0.20371106266975403, 0.8342629075050354, 0.24124501645565033, 1.3615126609802246, -0.23512592911720276, 2.073133, -1.0, -0.9773573279380798, -0.7824113368988037, 0.8318026661872864, -0.9848536849021912, -0.552554726600647, -0.28161293268203735, -0.3877178430557251, 0.2742151916027069, -0.5137662887573242, 0.6352485418319702, -0.7182226181030273, -0.4763774275779724, -0.09087696671485901, -0.200660839676857, -1.0, -0.07952868938446045, -0.33128654956817627, 0.017482861876487732, 0.10041153430938721, -1.0, -0.5918828248977661, -0.5234440565109253, 0.7525928020477295, -0.47376275062561035, 0.9241651296615601, -0.7082985639572144, -0.3461708426475525, -0.19287168979644775, -0.7628486752510071, -0.15316364169120789, -0.18589895963668823, 0.3216366469860077, 1.9238429069519043, -0.17163486778736115, 1.4298486709594727, -1.0, -0.35596904158592224, -0.3914569020271301], "live_pnl": -0.4905485212802887, "live_wins": 21, "live_total": 60, "age": 98, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.7784264497866323, "backtest_acc": 0.5478424015009381, "backtest_n": 4079}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement relative to order flow and book friction.
    """
    df = df.copy()

    # 1. Price Velocity: How much the price is moving per second remaining.
    # Normalizes the speed of price action.
    df['price_velocity'] = df['price_vs_strike_pct'] / (df['seconds_remaining'] + 1e-6)

    # 2. Flow Efficiency: The ratio of price movement to aggressive delta.
    # High efficiency = low liquidity/slippage; Low efficiency = high absorption.
    df['flow_efficiency'] = abs(df['price_velocity']) / (abs(df['mean_delta']) + 1e-6)

    # 3. Microstructure Friction: Combined measure of book and price dispersion.
    # High friction indicates a noisy, wide market.
    df['microstructure_friction'] = df['obi_spread'] + df['price_spread']

    # 4. Delta-OBI Coherence: Do the aggressors and the book agree?
    df['delta_obi_coherence'] = df['mean_delta'] * df['mean_obi']

    # Drop metadata and target as per instructions
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Dynamic Risk-Adjusted Hurdle (DRAH)".

    Logic:
    1. Calculate Dynamic Hurdle: 
       The required EV increases based on:
       - Base hurdle (3%)
       - Volatility (higher vol = higher hurdle to protect Sharpe)
       - Time Decay (as seconds_remaining -> 0, the hurdle rises because 
         there is less time for the signal to realize).
    2. Execution:
       - Only trade if the best EV exceeds the dynamic hurdle.
       - Scale bet size by:
         a) The magnitude of the edge relative to the hurdle.
         b) The inverse of microstructure friction (spreads).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    best_ev = max(ev_up, ev_down)
    side = 'Up' if ev_up > ev_down else 'Down'

    # Extract engineered features
    friction = features_row.get('microstructure_friction', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)
    
    # 1. Calculate Dynamic Hurdle
    # Base hurdle is 3%. 
    # Volatility adds up to 2% to the hurdle.
    # Time decay adds up to 3% as the window closes.
    base_hurdle = 0.03
    vol_adjustment = min(0.02, mean_vol * 0.1) 
    time_adjustment = max(0.0, (300 - seconds_remaining) / 300 * 0.03)
    
    total_hurdle = base_hurdle + vol_adjustment + time_adjustment

    # 2. Decision and Sizing
    if best_ev > total_hurdle:
        # Scale bet fraction:
        # - Higher edge relative to hurdle = larger bet.
        # - Higher friction (spreads) = smaller bet.
        # - We cap the bet at 1.0.
        
        edge_multiplier = best_ev / total_hurdle
        friction_penalty = 1.0 / (1.0 + friction)
        
        bet_fraction = edge_multiplier * friction_penalty
        
        # Ensure bet is within reasonable bounds [0.05, 1.0]
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0