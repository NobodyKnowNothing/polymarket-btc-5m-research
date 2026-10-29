# Var 41 | BT: -0.444 | Live PnL: $-15.24 | Trades: 61
# Desc: Implements "Strike-Proximity Coherence & Absorption Capture (SPCAC)," which uses the proximity to the strike price to dynamically adjust risk thresholds and targets "coiled spring" breakouts where ord
# Stats: {"live_returns": [-0.2395089864730835, -0.08984304964542389, 0.2662493586540222, -0.5009049773216248, 0.18909376859664917, -0.2873324453830719, -0.6160944104194641, 0.4922892153263092, -0.9374092817306519, -1.0, -0.17152750492095947, 0.06343662738800049, 0.5135553479194641, -0.7034030556678772, -0.7378471493721008, -1.0, 0.7947274446487427, -0.3297290802001953, -0.9030486345291138, -0.33669665455818176, -0.31539782881736755, -1.0, -0.2629023492336273, -0.4027267098426819, -0.3321586847305298, -0.11365661025047302, -0.8265224695205688, 0.40838944911956787, -1.0, -0.1961444616317749, 2.453085422515869, -0.4397770166397095, -0.40684619545936584, -0.5173882842063904, -0.26885801553726196, 0.01589284837245941, -0.6181338429450989, 0.7654932737350464, 0.1513517200946808, -0.13235168159008026, -0.23226915299892426, 0.1463533490896225, -0.7524589896202087, -0.2638317346572876, -0.4814320504665375, -0.07993512600660324, -0.2706690728664398, -0.40873223543167114, 0.47929486632347107, -0.28068527579307556, -0.4553258419036865, -0.9132899045944214, -0.9113191962242126, -0.6996320486068726, -0.18690483272075653, -0.6165722012519836, -0.3365890383720398, -0.1858711540699005, -0.445600688457489, 0.4735009968280792, -0.24504296481609344], "live_pnl": -15.239655494689941, "live_wins": 14, "live_total": 61, "age": 95, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.44393967196865985, "backtest_acc": 0.5437219730941704, "backtest_n": 2122}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    coherence between order book pressure and aggressive flow, as well as 
    the efficiency of price movement relative to that flow.
    """
    X = df.copy()

    # 1. OBI-Delta Coherence: Are the book and the aggressive traders in sync?
    # High positive = both bullish; High negative = both bearish.
    X['obi_delta_coherence'] = X['mean_obi'] * X['mean_delta']

    # 2. Delta-Price Efficiency: How much price moves per unit of delta.
    # Low values suggest "absorption" (heavy walls); high values suggest "vacuum" (easy moves).
    X['delta_price_efficiency'] = X['mean_delta'] / (X['price_std'] + 1e-6)

    # 3. Cross-Exchange Tension: Disagreement between exchanges relative to volatility.
    X['cross_exchange_tension'] = X['price_spread'] / (X['mean_volatility'] + 1e-6)

    # 4. Momentum-OBI Alignment: Is the trend supported by the book?
    X['momentum_obi_sync'] = X['mean_momentum'] * X['mean_obi']

    # Drop metadata and target columns to ensure no leakage
    # Note: seconds_remaining is passed separately to strategy_filter
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

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
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Strike-Proximity Coherence & Absorption Capture (SPCAC)".
    
    This strategy adapts its hurdle rate based on how close the price is to the strike.
    Near the strike, it seeks 'coiled spring' setups (high coherence, low efficiency).
    Away from the strike, it seeks 'trend confirmation' (high coherence, high efficiency).
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

    # 2. Extract Microstructure Metrics
    coherence = features_row.get('obi_delta_coherence', 0.0)
    efficiency = features_row.get('delta_price_efficiency', 0.0)
    tension = features_row.get('cross_exchange_tension', 0.0)
    proximity = abs(features_row.get('price_vs_strike_pct', 0.0))
    mean_delta = features_row.get('mean_delta', 0.0)

    # 3. Dynamic Regime Logic
    multiplier = 1.0
    
    # A) Near-Strike Regime (High Volatility/Uncertainty)
    # Price is within 0.15% of strike. We require higher conviction.
    if proximity < 0.0015:
        hurdle = 0.04 
        # Target "Coiled Spring": High coherence + High absorption (low efficiency)
        if abs(coherence) > 0.05 and abs(efficiency) < 0.5:
            multiplier = 1.4
        else:
            multiplier = 0.6 # Reduce exposure in chaotic near-strike noise
            
    # B) Out-of-the-Money/In-the-Money Regime (Trend Following)
    # Price is further from strike. We look for momentum confirmation.
    else:
        hurdle = 0.025
        # Target "Trend Ignition": High coherence + High efficiency (easy movement)
        if abs(coherence) > 0.02 and abs(efficiency) > 1.0:
            multiplier = 1.2
        else:
            multiplier = 0.8

    # 4. Global Safety Filters
    # If exchanges are in high tension (disagreement), scale down significantly
    if tension > 2.5:
        multiplier *= 0.4
        
    # If the delta direction contradicts our side, we are fighting the flow
    if (side == 'Up' and mean_delta < 0) or (side == 'Down' and mean_delta > 0):
        multiplier *= 0.5

    # 5. Execution & Sizing
    if best_ev < hurdle or multiplier <= 0:
        return None, 0.0

    # Time-decay: Reduce bet size as the 5-minute window expires to avoid last-second whipsaws
    time_factor = (seconds_remaining / 300.0)
    
    # Final bet fraction: Scale by EV relative to hurdle and apply regime multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold for profitability
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction