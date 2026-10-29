# Var 156 | BT: -0.315 | Live PnL: $-0.29 | Trades: 60
# Desc: Implements "Signal-to-Noise Ratio (SNR) Adaptive Execution," which scales the required Expected Value (EV) hurdle and the bet size based on the clarity of the microstructure (the alignment of Order Bo
# Stats: {"live_returns": [0.002442800000000002, 0.006331500000000004, 0.2226821631193161, -0.06339184194803238, -0.10568176209926605, 0.06698185000000001, -0.09755556285381317, 0.06663323938846588, -0.1753253936767578, -0.11461666971445084, -0.1503494530916214, 0.14366115629673004, -0.05, 0.006331500000000004, 0.0801457092165947, 0.32128870487213135, 0.09895705, -0.05, 0.08070164918899536, 0.5086694955825806, -0.05, -0.06090601161122322, -0.5022284984588623, 0.0903876, 0.0116128, -0.11216816306114197, -0.06307545304298401, -0.08334260433912277, 0.05462775, -0.09884452819824219, 0.21546494960784912, -0.05, -0.05469130352139473, -0.05, -0.05, -0.05531303584575653, -0.0550859272480011, -0.05, 0.006331500000000004, 0.015074856579303741, -0.09146106988191605, -0.05281079187989235, 0.0116128, 0.1607828140258789, 0.0029626999999999987, -0.056077077984809875, 0.22580385208129883, -0.08686210215091705, -0.14330342411994934, -0.05, -0.20519357919692993, -0.0760512501001358, 0.005738350000000003, -0.05607207491993904, 0.1375778317451477, 0.001933749999999998, 0.003609880805015564, -0.15314429998397827, 0.3244071304798126, -0.05], "live_pnl": -0.29079654812812805, "live_wins": 27, "live_total": 60, "age": 97, "rounds_underwater": 13, "abstain_rounds": 0, "backtest_score": -0.31486567828992756, "backtest_acc": 0.5965076980848667, "backtest_n": 4423}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    creating a Signal-to-Noise Ratio (SNR) metric that captures the 
    reliability of the current microstructure state.
    """
    df = df.copy()

    # 1. Microstructure Signal: Combined magnitude of intent (OBI) and action (Delta)
    # We use absolute values to represent the total 'energy' in the book/flow.
    df['micro_signal'] = df['mean_delta'].abs() + df['mean_obi'].abs()

    # 2. Microstructure Noise: Combined spread of price and order book
    # Higher values indicate less certainty in the current mid-price/intent.
    df['micro_noise'] = df['price_spread'] + df['obi_spread'] + 1e-6

    # 3. Alignment Penalty: Penalize when OBI and Delta are fighting each other
    # (e.g., OBI is positive/bids high, but Delta is negative/aggressive selling).
    # This is a key indicator of absorption or high-friction environments.
    df['alignment_conflict'] = (np.sign(df['mean_delta']) * np.sign(df['mean_obi']) < 0).astype(float)

    # 4. SNR: The core feature. Signal divided by noise, penalized by conflict.
    # High SNR = Clear, high-conviction direction.
    # Low SNR = Noisy, conflicting, or high-spread environment.
    df['snr'] = df['micro_signal'] / (df['micro_noise'] * (1.0 + df['alignment_conflict']))

    # 5. Price-Flow Elasticity: How much price moves per unit of delta.
    # High elasticity = "Vacuum" (price moves easily); Low elasticity = "Wall" (price is stuck).
    df['price_elasticity'] = df['price_vs_strike_pct'].abs() / (df['mean_delta'].abs() + 1e-6)

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
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "SNR Adaptive Execution."

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Calculate a Dynamic Hurdle: The required EV increases as the 
       microstructure SNR decreases. In noisy markets, we demand more edge.
    3. Scale Bet Size: Bet size is proportional to the edge, but heavily 
       capped by the SNR. We only place full bets when the signal is crystal clear.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine the best available edge
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    snr = features_row.get('snr', 0.0)
    
    # 1. Dynamic Hurdle: 
    # Base hurdle of 2.5% EV, but decays/increases based on SNR.
    # If SNR is 0, hurdle is high. If SNR is high, hurdle approaches 2.5%.
    # This prevents "trading the noise" in low-conviction environments.
    dynamic_hurdle = 0.025 / (1.0 + (snr * 0.5))

    if best_ev > dynamic_hurdle:
        # 2. Adaptive Bet Sizing:
        # Scale bet by:
        # a) Magnitude of edge (target 4% EV for full bet)
        # b) Signal Quality (SNR): We use a saturation function to scale bet.
        #    A high SNR of 4.0+ results in a multiplier near 1.0.
        
        edge_multiplier = best_ev / 0.04
        snr_multiplier = min(1.0, snr / 4.0)
        
        # 3. Time Decay:
        # As we approach the 5-minute mark, reduce size to avoid last-second volatility.
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)
        
        bet_fraction = edge_multiplier * snr_multiplier * time_factor
        
        # Clamp bet_fraction between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0