# Var 51 | BT: -0.566 | Live PnL: $-5.67 | Trades: 61
# Desc: Implements "Asymmetric Squeeze & Lead-Lag Dislocation (ASLLD)", which uses funding-delta mismatches to identify squeeze regimes and cross-exchange price gaps to identify lead-lag opportunities, applyi
# Stats: {"live_returns": [-1.0, -0.5427395701408386, 0.4346257150173187, 0.058726727962493896, -0.16906341910362244, -1.0, -1.0, -1.0, -0.46738767623901367, 1.1965441703796387, -0.30211129784584045, -0.5991939306259155, -0.0730302706360817, 0.022210359573364258, -0.2445172220468521, -0.6450248956680298, 1.3885104656219482, -0.31490060687065125, 4.245841, -1.0, -0.9677037596702576, -1.0, 1.5227041244506836, 1.10764479637146, -1.0, -0.4785330295562744, -0.4622677266597748, -1.0, -0.27994316816329956, 0.05525299906730652, -0.49634334444999695, 4.550646, -0.32972389459609985, 3.1194439999999997, -0.34163352847099304, -0.923016369342804, -0.4560539424419403, -1.0, -0.32913872599601746, -0.8205283284187317, -0.6436150074005127, -0.24960362911224365, -0.19124779105186462, 2.39541, 0.021783113479614258, 0.1962229609489441, -0.19109536707401276, -0.45775192975997925, 0.46656492352485657, 0.6263329982757568, -1.0, -1.0, -0.4952118694782257, -1.0, -1.0, -0.6125759482383728, -1.0, 1.0616087913513184, -0.430108904838562, -0.6892392635345459, -0.9331790208816528], "live_pnl": -5.666410446166992, "live_wins": 17, "live_total": 61, "age": 77, "rounds_underwater": 10, "abstain_rounds": 0, "backtest_score": -0.5662664773914988, "backtest_acc": 0.5753571428571429, "backtest_n": 2469}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    funding-delta mismatches (squeezes) and cross-exchange price dislocations.
    """
    df = df.copy()

    # 1. Squeeze Signal: The product of funding and delta. 
    # A negative value indicates a potential squeeze (e.g., positive funding but negative delta).
    df['funding_delta_mismatch'] = df['mean_funding'] * df['mean_delta']

    # 2. Lead-Lag Dislocation: How much is Binance leading the average price?
    # We use the average of the three exchanges as the 'fair' baseline.
    avg_price = (df['binance_price'] + df['coinbase_price'] + df['kraken_price']) / 3.0
    df['price_dislocation'] = df['binance_price'] - avg_price

    # 3. OBI Acceleration: The rate of change in order book imbalance.
    # Grouping by market_slug to ensure we only diff within the same 5-minute window.
    df['obi_acceleration'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)

    # 4. Volatility-Normalized Flow: Delta relative to the current volatility regime.
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to combat the 50/50 noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Asymmetric Squeeze & Lead-Lag Dislocation (ASLLD)".
    This strategy treats the model's prediction as a base signal and uses microstructure 
    regimes (squeezes and dislocations) to either massively amplify or heavily veto the bet.
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

    # 2. Base Hurdle: We need a minimum edge to overcome noise.
    hurdle = 0.025
    if best_ev < hurdle:
        return None, 0.0

    # 3. Extract Engineered Metrics
    mismatch = features_row.get('funding_delta_mismatch', 0.0)
    dislocation = features_row.get('price_dislocation', 0.0)
    mean_f = features_row.get('mean_funding', 0.0)
    mean_d = features_row.get('mean_delta', 0.0)
    price_spread = features_row.get('price_spread', 0.0)

    multiplier = 1.0

    # 4. Squeeze Regime Logic (High Alpha/High Risk)
    # A squeeze occurs when funding and aggressive delta are in opposition.
    squeeze_dir = None
    if mean_f > 0.0001 and mean_d < -0.005:
        squeeze_dir = 'Down'  # Long squeeze
    elif mean_f < -0.0001 and mean_d > 0.005:
        squeeze_dir = 'Up'    # Short squeeze

    if squeeze_dir:
        if side == squeeze_dir:
            # The model correctly identified the squeeze direction
            multiplier *= 1.6
        else:
            # The model is predicting a trend that is actually a squeeze reversal
            multiplier *= 0.1

    # 5. Lead-Lag Dislocation Logic
    # If Binance is significantly leading the other exchanges, it's a high-conviction signal.
    if abs(dislocation) > 0.0005:
        dislocation_dir = 'Up' if dislocation > 0 else 'Down'
        if side == dislocation_dir:
            multiplier *= 1.3
        else:
            multiplier *= 0.5

    # 6. Uncertainty Vetoes
    # High cross-exchange price spread suggests high uncertainty/noise.
    if price_spread > 0.001:
        multiplier *= 0.4

    # 7. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes.
    time_factor = max(0.1, seconds_remaining / 300.0)

    # Scale bet_fraction by (Edge / Hurdle) * regime_multiplier * time_decay.
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Final safety check
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction