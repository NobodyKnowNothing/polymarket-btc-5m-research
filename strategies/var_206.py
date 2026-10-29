# Var 206 | BT: -0.121 | Live PnL: $-7.55 | Trades: 60
# Desc: Implements "Lead-Venue Efficiency Breakout (LVEB)," which identifies moments when Binance's price movement per unit of aggressive delta (Delta-Price Efficiency) significantly exceeds the cross-exchang
# Stats: {"live_returns": [-0.9947228954870836, -0.9814218814618009, 1.9685374793918555, -0.9975217424580203, 0.12250244617462158, 1.8827911709582659, -0.9935876667297747, 1.5039226506503116, -0.9872622005869459, 1.230917244363504, -0.9844659745923252, -0.9832968773064625, -0.9911704312945157, -0.9963689597700017, -0.9889389399353352, 1.8473212595416708, -0.9967705362692678, -0.9921203802551984, 2.6367222553738037, -0.9982411095653665, -0.9971388171457323, -0.9915196768573129, 1.1377344131469727, -0.9767516493521134, -0.998367272229065, -0.9912222873919204, 0.0792161282838485, -0.9919569057628915, -0.9891219403690087, -0.9886993473469499, 0.06798697116527719, 2.376424251295154, -0.999257633828835, -1.0, -0.9988285558093174, -0.9947038160976356, -0.9943104857341526, -0.9980677883570521, 0.044210970401763916, 2.15343597472687, -0.9931301165683775, -0.9957274165193498, -0.9901736449391404, -0.9898042353636705, 1.796348615612704, 1.4548815119579706, -0.9789562768459765, 0.1206369400024414, -0.999166495243885, -0.9769412191873463, 1.6461287562442286, 1.3226183934024038, -0.9915034104763186, 2.505640056471493, 2.6442106132492254, -0.994559540597995, -0.9998187953432042, -0.9939014232604653, 0.16085699750859128, 1.4503514199919645], "live_pnl": -7.546119689941406, "live_wins": 22, "live_total": 60, "age": 105, "rounds_underwater": 49, "abstain_rounds": 0, "backtest_score": -0.12117975437990479, "backtest_acc": 0.5574087078651685, "backtest_n": 4187}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'efficiency' of price movement relative to delta across exchanges.
    """
    df = df.copy()

    # 1. Calculate Price Change % for each exchange relative to strike
    # We use the provided strike_price to find the displacement
    df['binance_pc'] = (df['binance_price'] - df['strike_price']) / df['strike_price']
    df['coinbase_pc'] = (df['coinbase_price'] - df['strike_price']) / df['strike_price']
    df['kraken_pc'] = (df['kraken_price'] - df['strike_price']) / df['strike_price']

    # 2. Delta-Price Efficiency (DPE): How much price moves per unit of delta
    # High DPE = Low friction/Liquidity gap (Price moves easily)
    # Low DPE = High friction/Absorption (Price is being fought)
    eps = 1e-6
    df['binance_dpe'] = abs(df['binance_pc']) / (abs(df['binance_delta']) + eps)
    df['coinbase_dpe'] = abs(df['coinbase_pc']) / (abs(df['coinbase_delta']) + eps)
    df['kraken_dpe'] = abs(df['kraken_pc']) / (abs(df['kraken_delta']) + eps)

    # 3. Cross-Exchange Efficiency Metrics
    df['mean_dpe'] = (df['binance_dpe'] + df['coinbase_dpe'] + df['kraken_dpe']) / 3.0
    # Ratio of Binance efficiency to the market average
    df['binance_dpe_ratio'] = df['binance_dpe'] / (df['mean_dpe'] + eps)

    # 4. Momentum and Acceleration
    df['binance_delta_accel'] = df.groupby('market_slug')['binance_delta'].diff().fillna(0)
    
    # 5. Standardize/Clean
    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Replace inf and NaN
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to capture microstructure efficiency signals."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Venue Efficiency Breakout (LVEB)".

    Logic:
    1. Identify Breakout: If Binance's DPE is significantly higher than the 
       cross-exchange average (binance_dpe_ratio > 1.4), it suggests a 
       liquidity gap is being exploited on the lead exchange.
    2. Regime-Based Hurdle:
       - Breakout Regime: Lower EV hurdle (0.02) because the move is "clean" 
         and likely to continue.
       - Normal Regime: Higher EV hurdle (0.04) to ensure edge.
    3. Directional Alignment: In a breakout, follow the sign of Binance Delta.
    4. Dynamic Sizing: Scale by EV and penalize for price extension from strike.
    """
    # Exit if too little time remains
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    binance_delta = features_row.get('binance_delta', 0.0)
    dpe_ratio = features_row.get('binance_dpe_ratio', 1.0)
    price_dist = abs(features_row.get('price_vs_strike_pct', 0.0))

    # Detect Breakout Regime
    is_breakout = dpe_ratio > 1.4

    side = None
    current_ev = 0.0

    if is_breakout:
        # In a breakout, we follow the direction of the lead-venue delta
        if binance_delta > 0 and ev_up > 0.02:
            side = 'Up'
            current_ev = ev_up
        elif binance_delta < 0 and ev_down > 0.02:
            side = 'Down'
            current_ev = ev_down
    else:
        # Normal regime: standard EV comparison
        if ev_up > ev_down and ev_up > 0.04:
            side = 'Up'
            current_ev = ev_up
        elif ev_down > ev_up and ev_down > 0.04:
            side = 'Down'
            current_ev = ev_down

    if side is None:
        return None, 0.0

    # --- Dynamic Bet Sizing ---
    # Base size on how much the EV exceeds the hurdle
    # Maximize size when EV is high, but scale down as price gets extended
    # (Avoid buying the tail end of a breakout)
    
    # Scale factor based on EV magnitude (normalized to 0.05)
    ev_scale = min(1.0, current_ev / 0.05)
    
    # Extension penalty: as price moves away from strike, reduce size
    # A 1% move (0.01) reduces size significantly
    extension_penalty = 1.0 / (1.0 + price_dist * 30.0)
    
    # Breakout boost: if it's a real breakout, we can be slightly more aggressive
    breakout_boost = 1.2 if is_breakout else 1.0

    bet_fraction = ev_scale * extension_penalty * breakout_boost

    # Final safety clamp
    return side, min(1.0, max(0.05, bet_fraction))