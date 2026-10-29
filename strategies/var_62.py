# Var 62 | BT: 0.226 | Live PnL: $-8.76 | Trades: 60
# Desc: Implements "Inter-Exchange Lead-Lag & Flow-Confluence (IILLFC)," which targets high-alpha windows where Binance price velocity diverges from other exchanges, validated by aggressive flow and order boo
# Stats: {"live_returns": [1.6554160000000002, 0.05287247896194458, -1.0, 3.1194439999999997, -1.0, -0.4061813950538635, 0.06070351600646973, -0.40081650018692017, -1.0, -1.0, -1.0, -0.7228974103927612, -1.0, 2.651597, -1.0, -1.0, -1.0, -0.5241458415985107, -1.0, -1.0, 4.550646, -0.6088364124298096, -0.5991410613059998, 2.795066, -1.0, 0.9830037951469421, -0.3973660171031952, -0.7877582907676697, -0.8114525675773621, -0.8330861330032349, -0.9269956350326538, -0.8916897177696228, -1.0, 0.029406815767288208, -0.4947056472301483, 0.01862657070159912, 0.011613219976425171, -1.0, -0.6055342555046082, -1.0, -0.5946028828620911, -1.0, 0.07215982675552368, -1.0, -1.0, -1.0, -0.7716278433799744, 0.033524155616760254, 1.6210253238677979, -1.0, 0.2755154073238373, -1.0, 2.39541, 0.5249868631362915, -0.767824649810791, 0.0194699764251709, 1.0208759307861328, 1.890848, 0.6031712293624878, -1.0], "live_pnl": -8.759278297424316, "live_wins": 21, "live_total": 60, "age": 320, "rounds_underwater": 33, "abstain_rounds": 1, "backtest_score": 0.22620825183551266, "backtest_acc": 0.5414285714285715, "backtest_n": 618}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    detecting when Binance leads the market and if flow validates that lead.
    """
    df = df.copy()

    # 1. Price Velocity and Divergence
    # Calculate velocity for Binance and the market average
    df['binance_price_vel'] = df.groupby('market_slug')['binance_price'].pct_change()
    
    # Calculate mean price across exchanges to find the 'market' velocity
    df['avg_price'] = df[['binance_price', 'coinbase_price', 'kraken_price']].mean(axis=1)
    df['mean_price_vel'] = df.groupby('market_slug')['avg_price'].pct_change()
    
    # Price Divergence: Is Binance moving faster/differently than the consensus?
    df['price_divergence'] = df['binance_price_vel'] - df['mean_price_vel']

    # 2. Flow Confluence
    # Check if the aggressive trade direction (delta) matches the book imbalance (OBI)
    # 1.0 = perfect alignment, -1.0 = total divergence (absorption), 0.0 = neutral
    df['flow_confluence'] = np.sign(df['binance_obi']) * np.sign(df['binance_delta'])

    # 3. Delta Acceleration
    # Is the aggressive buying/selling accelerating?
    df['delta_accel'] = df.groupby('market_slug')['binance_delta'].diff()

    # Drop metadata and temporary columns
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'avg_price']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs from pct_change/diff and infinities
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Inter-Exchange Lead-Lag & Flow-Confluence (IILLFC)".
    A trade is valid only if:
    1. The ML model sees EV.
    2. Binance is leading the price movement (Price Divergence).
    3. The flow (Delta/OBI) confirms the direction (Flow Confluence).
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

    # 2. Extract Microstructure Signals
    # How much is Binance leading the consensus?
    lead_signal = features_row.get('price_divergence', 0.0)
    # Is the flow supporting the move?
    flow_signal = features_row.get('flow_confluence', 0.0)

    # 3. Directional Validation
    # If we want to go 'Up', Binance must be leading the price up (lead_signal > 0)
    # If we want to go 'Down', Binance must be leading the price down (lead_signal < 0)
    direction_match = (side == 'Up' and lead_signal > 0) or (side == 'Down' and lead_signal < 0)
    
    # 4. Entry Gating
    # Thresholds
    base_hurdle = 0.03
    
    # Requirement A: Must have positive EV and direction match
    if best_ev < base_hurdle or not direction_match:
        return None, 0.0
    
    # Requirement B: Flow must be aligned (no absorption)
    if flow_signal <= 0:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale multiplier based on how strong the Binance lead is
    # We normalize lead_signal (typically very small) by a factor of 0.001
    lead_multiplier = 1.0 + min(1.0, abs(lead_signal) / 0.001)
    
    # Scale by edge strength relative to hurdle
    edge_ratio = best_ev / base_hurdle
    
    # Time decay: reduce exposure as we approach the 5-min resolution
    time_factor = max(0.2, seconds_remaining / 300.0)

    # Final bet fraction: (Edge) * (Lead Strength) * (Flow Alignment) * (Safety Coeff)
    # flow_signal is 1.0 here because we already gated for <= 0
    bet_fraction = edge_ratio * lead_multiplier * flow_signal * 0.5
    bet_fraction *= time_factor

    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to avoid dust
    if bet_fraction < 0.1:
        return None, 0.0

    return side, bet_fraction