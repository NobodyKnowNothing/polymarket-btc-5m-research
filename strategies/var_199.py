# Var 199 | BT: 0.890 | Live PnL: $91.38 | Trades: 64
# Desc: Targets the "Absorption-to-Impact" transition by detecting spikes in price-movement efficiency relative to aggressive flow, signaling the moment liquidity walls are breached.
# Stats: {"live_returns": [-1.0, -1.0, 0.09182599999999996, 0.05925399999999992, -1.0, 0.08073200000000003, -1.0, 0.1233902, 0.09182599999999996, 0.16389999999999993, 0.11476700000000006, -1.0, 0.5381629712069902, -1.0, -0.470242121402235, -0.13360649577959083, 0.10316799999999993, 0.04431889141379719, -1.0, -1.0, -1.0, 0.038675000000000015, 0.16389999999999993, 0.10260386698889445, -0.9440575500605897, -1.0, 1.1289913080237381, 0.11476700000000006, 0.028705999999999898, -0.7366495646397666, 0.028705999999999898, -0.19265472499376468, 92.346277, -0.1982815592245335, 1.233902, 0.02685334437313011, -0.9760774580472966, 0.05925399999999992, -1.0, -0.35305105799484315, -1.0, 2.39541, -0.653341734841516, 2.795066, -1.0, 0.05679755564791633, -1.0, -1.0, -0.2866855638023358, 0.10316799999999993, 1.890848, -1.0, -1.0, 0.5195178596014366, -0.7464330038888329, 3.9731249999999996, -0.27561992974063476, -0.2553137352748647, 0.608801063277578, -0.3774551106090147, -1.0, 4.550646, 2.39541, 0.009909556886087856], "live_pnl": 91.38321000711973, "live_wins": 32, "live_total": 64, "age": 812, "rounds_underwater": 0, "abstain_rounds": 24, "backtest_score": 0.8901479948480817, "backtest_acc": 0.5545905707196029, "backtest_n": 378}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the transition from liquidity absorption to price impact (Efficiency Acceleration).
    """
    df = df.copy()

    # 1. Price Change (within the market window)
    df['price_diff'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Delta Efficiency: How much price moves per unit of aggressive delta
    # We use log1p to stabilize the ratio and handle the wide range of delta values
    df['delta_efficiency'] = np.log1p(df['price_diff'].abs() / (df['sum_delta'].abs() + 1e-6))
    
    # 3. Smooth Efficiency to reduce microstructure noise
    df['delta_efficiency_smooth'] = df.groupby('market_slug')['delta_efficiency'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # 4. Efficiency Acceleration: The "Breakout" signal
    # A sudden spike in efficiency suggests the market has moved from an absorption 
    # regime (high delta, low price move) to an impact regime (delta starts moving price).
    df['eff_accel'] = df.groupby('market_slug')['delta_efficiency_smooth'].diff().fillna(0)

    # 5. OBI-Efficiency Alignment
    df['obi_eff_sync'] = df['mean_obi'] * df['delta_efficiency_smooth']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for detecting regime transitions."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Absorption-Breakout Transition (ABT)" logic.

    Logic:
    1. Detect Efficiency Spike: 
       Look for a positive 'eff_accel' (the moment price starts moving efficiently 
       relative to the volume being traded).
    2. Directional Consensus:
       Ensure the aggressive flow (delta) and order book intent (OBI) are aligned.
    3. Proximity Filter:
       Avoid entering if the price is already too far from the strike (preventing 
       chasing exhausted moves).
    4. Dynamic Sizing:
       Scale the bet based on the magnitude of the efficiency acceleration.
    """
    # Exit if too little time remains
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    eff_accel = features_row.get('eff_accel', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)
    price_vs_strike_pct = features_row.get('price_vs_strike_pct', 0.0)

    side = None
    bet_fraction = 0.0

    # Threshold for a "significant" efficiency spike, adjusted by volatility
    # In high volatility, we require a larger spike to confirm a breakout.
    trigger_threshold = 0.01 * (1.0 + mean_vol)

    if eff_accel > trigger_threshold:
        # Check for directional consensus (Delta and OBI agree)
        direction_consensus = (mean_delta > 0 and mean_obi > 0) or (mean_delta < 0 and mean_obi < 0)
        
        # Check if we are too far from the strike (avoiding the "top" or "bottom")
        # If price is > 0.7% away from strike, the move might be overextended.
        not_extended = abs(price_vs_strike_pct) < 0.007

        if direction_consensus and not_extended:
            # Check for positive EV on the side of the consensus
            if mean_delta > 0 and ev_up > 0.02:
                side = 'Up'
                # Scale bet by the intensity of the efficiency spike
                bet_fraction = min(1.0, eff_accel * 3.0)
            elif mean_delta < 0 and ev_down > 0.02:
                side = 'Down'
                bet_fraction = min(1.0, eff_accel * 3.0)

    # Final safety clamp
    if side:
        return side, max(0.1, min(1.0, bet_fraction))

    return None, 0.0