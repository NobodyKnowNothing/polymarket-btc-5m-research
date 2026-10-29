# Var 208 | BT: -1.100 | Live PnL: $-2.04 | Trades: 60
# Desc: Implements "Volatility-Normalized Impact-Efficiency Scaling (VRIES)," which scales the required expected value by the current volatility regime and penalizes trades occurring in "liquidity vacuums" (h
# Stats: {"live_returns": [-0.02, -1.0, 0.0025325999999999994, -0.02, 0.006781042651408199, 0.024678040000000002, 0.015335339918203683, 0.0018365199999999977, 0.0032779999999999997, -0.16312977521327882, 0.0007734999999999999, -1.0, 0.0074289292097091675, -0.02, -0.39670367563118913, -1.0, 0.0008114427328109741, 0.011911094928294369, 2.39541, -0.22820291269001167, 0.0007734999999999999, -0.025089009373444205, -0.1279505948003859, 0.20678414238412215, -0.1356029977436054, -0.02, -0.02614866823111729, 0.0013201441464244061, 1.17539206244912, -0.02, -0.07482003012933303, -0.38702210450007307, 0.0025325999999999994, -0.05028080940246582, 0.0038338638842105865, 0.09859864542703334, -0.021330800549202434, 0.09241745676398744, -0.09847870962447561, 0.08335231390236747, -0.06888993971247265, -0.02, 1.4562249999999999, 0.055803225989874305, -0.02, -0.6998739129405588, -0.06380464463032275, -0.17266468899590823, 0.0015012392689362497, 0.07011250000000001, 0.019412761433176273, 0.00018748000000000029, -0.5185848947981906, -0.02, -0.02, 0.35456500318923045, -0.13657921029519926, -0.4029031772318466, -1.0, -0.15262325068338284], "live_pnl": -2.037095069885254, "live_wins": 27, "live_total": 60, "age": 114, "rounds_underwater": 14, "abstain_rounds": 0, "backtest_score": -1.100015568178007, "backtest_acc": 0.5476066144473455, "backtest_n": 3760}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    volatility-adjusted flow and the efficiency of price impact.
    """
    df = df.copy()

    # 1. Microstructure Momentum: Price change since last snapshot
    # We use binance_price as the primary reference for price action
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Impact Efficiency: How much price moves per unit of aggressive delta
    # High efficiency = "Vacuum" (price moves easily, low liquidity)
    # Low efficiency = "Impactful" (price moves heavily, high liquidity/absorption)
    # We add a small epsilon to prevent division by zero
    df['impact_efficiency'] = df['price_change'] / (df['mean_delta'].abs() + 1e-6)

    # 3. Volatility-Adjusted Delta: Is the flow significant relative to current noise?
    df['vol_adj_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 4. Delta-Price Coherence: Do aggressive trades actually result in price movement?
    df['delta_price_coherence'] = df['mean_delta'] * df['price_change']

    # Drop metadata, target, and raw price columns used for calculation
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'binance_price', 'coinbase_price', 'kraken_price']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust microstructure signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Normalized Impact-Efficiency Scaling (VRIES)".

    Logic:
    1. Volatility Normalization: Instead of a fixed EV hurdle, we use a hurdle 
       scaled by the current volatility. In high-vol regimes, we require much 
       higher absolute EV to justify a trade.
    2. Vacuum Avoidance: We calculate the 'impact efficiency'. If the price is 
       moving too easily (high efficiency/vacuum), we drastically reduce the 
       bet size because the move is likely unstable and lacks liquidity support.
    3. Dynamic Sizing: Bet size is a function of the normalized edge.
    """
    # Exit if too little time remains
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine the best side
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > ev_up:
        side = 'Down'
        best_ev = ev_down
    else:
        return None, 0.0

    # Extract engineered features
    vol = features_row.get('mean_volatility', 0.001)
    efficiency = abs(features_row.get('impact_efficiency', 0.0))

    # --- 1. Volatility-Normalized EV ---
    # We normalize the EV by the volatility. 
    # A 2% edge in a 0.1% vol environment is massive; in a 1% vol environment, it's noise.
    # We use a smoothing constant (0.005) to prevent division by near-zero.
    norm_ev = best_ev / (vol + 0.005)

    # Required normalized EV threshold
    # This effectively means: required_ev = 0.4 * (vol + 0.005)
    if norm_ev < 0.4:
        return None, 0.0

    # --- 2. Execution & Sizing ---
    # Base bet size scales with the strength of the normalized edge
    # We cap the base bet at 1.0
    base_bet = min(1.0, norm_ev * 0.4)

    # --- 3. Vacuum Penalty ---
    # If efficiency is extremely high, the price is moving in a liquidity vacuum.
    # We penalize the bet size to avoid catching "fake" breakouts.
    # penalty = 1 / (1 + efficiency). 
    # If efficiency is 0 (stable), penalty is 1.0. 
    # If efficiency is 10 (vacuum), penalty is ~0.09.
    efficiency_penalty = 1.0 / (1.0 + efficiency)
    
    final_bet = base_bet * efficiency_penalty

    # Final safety clamp: ensure we always bet at least a tiny amount if we pass the hurdle
    # but cap it at 1.0.
    return side, min(1.0, max(0.02, final_bet))