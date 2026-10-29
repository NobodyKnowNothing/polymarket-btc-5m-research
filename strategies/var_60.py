# Var 60 | BT: 0.160 | Live PnL: $-7.26 | Trades: 61
# Desc: Implements "Microstructure Absorption & Divergence Gating (MADG)," which moves away from trend-following "lag capture" to focus on "exhaustion" signals—specifically identifying when aggressive order f
# Stats: {"live_returns": [0.27827260644584656, 0.35703155355960126, -0.2349601916472117, -0.4697751593589783, -0.46942016124725344, 0.011971358975969931, 0.34218011809880533, 0.018943173105559347, -0.23455055753389994, -0.23488864103953044, -0.2020479961236318, -0.4694897143046061, -0.20181038816769917, -0.2348107413450877, -0.23414626955986023, -0.20069642742474875, -0.152347043355306, -0.23431994835535686, -0.4687512652079264, -0.2013228181997935, -0.4678892819086711, -0.2007435385386149, -0.16909525593121846, -0.23474132895469665, -0.13582340796788533, -0.23471757690111797, -0.2342094639937083, 1.0686229291041616, 0.7332969013498098, -0.23313257972399393, -0.23449114521344502, -0.2343747929732005, -0.23377887527147928, -0.23458491524060568, 0.5620125670095681, 0.013833093944178393, 0.3003527005248864, -0.46879590113957725, -0.2340938651561737, 0.6837683448182542, -0.23404829343159994, -0.4690309588114421, -0.23362609187761943, -0.2345601002375285, -0.23364493131637573, -0.23452260255813598, -0.2337394388516744, 0.46495296053991586, -0.4695502726236979, -0.4701131820678711, -0.23438842018445333, -0.23388267159461976, -0.2346628518899282, -0.23413288275400798, -0.23496803681055706, -0.23490457057952882, -0.23433566808700562, -0.2346468738714854, 0.013374814190820844, 0.47220424344762757, -0.1], "live_pnl": -7.261749734217804, "live_wins": 14, "live_total": 61, "age": 107, "rounds_underwater": 58, "abstain_rounds": 0, "backtest_score": 0.15965886150038391, "backtest_acc": 0.5246575342465754, "backtest_n": 2519}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    divergence between aggressive flow (delta) and passive liquidity (OBI).
    """
    df = df.copy()

    # 1. Divergence Features: Identifying Absorption
    # Positive product means delta and OBI are aligned (trend).
    # Negative product means delta and OBI are diverging (absorption/exhaustion).
    df['delta_obi_divergence'] = df['mean_delta'] * df['mean_obi']
    
    # 2. Volatility-Normalized Flow Intensity
    # Tells us if the delta is significant relative to the current market noise.
    df['vol_normalized_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # 3. Price Spread Acceleration
    # Measures if the disagreement between exchanges is widening or narrowing.
    df['price_spread_accel'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 4. Momentum-Volatility Ratio
    # Helps distinguish between high-conviction moves and noise.
    df['momentum_vol_ratio'] = df['mean_momentum'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
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
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Absorption & Divergence Gating (MADG)".
    
    Instead of just following the model's probability, we gate trades based on 
    microstructure exhaustion:
    1. Bullish Exhaustion: High aggressive buying (delta) but declining bids (OBI). 
       This is a 'trap'—we avoid 'Up' trades and look for 'Down' reversals.
    2. Bearish Exhaustion: High aggressive selling (delta) but rising bids (OBI). 
       This is a 'trap'—we avoid 'Down' trades and look for 'Up' reversals.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract key microstructure components
    m_delta = features_row.get('mean_delta', 0.0)
    m_obi = features_row.get('mean_obi', 0.0)
    
    # Define Exhaustion States
    # Bullish Exhaustion: Aggressive buying (m_delta > 0) hitting a wall of sellers (m_obi < 0)
    is_bull_exhaustion = m_delta > 0.02 and m_obi < -0.15
    # Bearish Exhaustion: Aggressive selling (m_delta < 0) hitting a wall of buyers (m_obi > 0)
    is_bear_exhaustion = m_delta < -0.02 and m_obi > 0.15

    # Base hurdle for any trade
    hurdle = 0.03
    
    # Decision Logic
    if ev_up > ev_down and ev_up > hurdle:
        # We want to go Up
        if is_bull_exhaustion:
            # The model says Up, but microstructure says we are hitting a wall. 
            # Abstain to avoid buying the top.
            return None, 0.0
        elif is_bear_exhaustion:
            # The model says Up, and microstructure confirms a reversal from selling exhaustion.
            # High conviction.
            side, confidence = 'Up', 1.0
        else:
            # Standard momentum/trend trade.
            side, confidence = 'Up', 0.5
            
    elif ev_down > ev_up and ev_down > hurdle:
        # We want to go Down
        if is_bear_exhaustion:
            # The model says Down, but microstructure says we are hitting a wall.
            # Abstain to avoid selling the bottom.
            return None, 0.0
        elif is_bull_exhaustion:
            # The model says Down, and microstructure confirms a reversal from buying exhaustion.
            # High conviction.
            side, confidence = 'Down', 1.0
        else:
            # Standard momentum/trend trade.
            side, confidence = 'Down', 0.5
    else:
        return None, 0.0

    # Time-decay factor: reduce exposure as the window approaches zero
    time_factor = max(0.2, seconds_remaining / 300.0)
    
    # Final bet sizing
    bet_fraction = confidence * time_factor
    
    # Safety check
    if bet_fraction < 0.1:
        return None, 0.0
        
    return side, min(1.0, bet_fraction)