# Var 4 | BT: -0.675 | Live PnL: $-1.42 | Trades: 60
# Desc: Implements "Synchronized Flow-Agreement Gating," which scales the model's edge by a multiplier derived from the alignment of momentum and delta, the magnitude of that flow, and the level of cross-exch
# Stats: {"live_returns": [-0.4592688417434692, -0.31640496850013733, -0.44000354409217834, 1.0588287115097046, -0.4584530178705851, -0.432429404258728, -0.3002541959285736, -0.46330663045247394, -0.29869595170021057, -0.24817167222499847, -0.46319501161575316, -0.2870666980743408, -0.3816753327846527, 0.7620687875580217, -0.39277178049087524, -0.45255733013153077, -0.24365323781967163, -0.27099835872650146, -0.41861411929130554, -0.4496928596496582, -0.2840104401111603, -0.2888016104698181, -0.22879555821418762, -0.46194168329238894, 0.870078980922699, -0.2740025520324707, -0.3355996310710907, -0.3753485083580017, -0.362900048494339, -0.327023446559906, -0.342383474111557, -0.4673871978123983, 0.43375590443611145, 3.1916783114481944, -0.39418184757232666, -0.2365230917930603, -0.46553491115570067, -0.2305527776479721, -0.2724672257900238, -0.2801416516304016, -0.2530210614204407, 0.017641037702560425, 0.7808893918991089, -0.14159028232097626, -0.3550858497619629, -0.2791307866573334, 0.7814902067184448, -0.2754790782928467, 1.602901577949524, -0.3938040328025818, 0.7399130463600159, -0.388029522895813, 0.009580835700035095, -0.26084795594215393, -0.27736470103263855, 0.4204714596271515, 1.5162016153335571, 2.109344170768752, -0.4689391525586446, -0.22170031865437825], "live_pnl": -1.424957513809204, "live_wins": 14, "live_total": 60, "age": 531, "rounds_underwater": 60, "abstain_rounds": 0, "backtest_score": -0.6753367338559291, "backtest_acc": 0.5609203416419731, "backtest_n": 1120}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features by calculating 
    flow synchronization, normalized flow magnitude, and exchange agreement.
    """
    X = df.copy()

    # 1. Flow Synchronization: Do momentum and delta point in the same direction?
    # We use the sign of mean_delta to see if momentum reinforces the aggressive flow.
    # Result is in [-1, 1]. 1 = perfect alignment, -1 = perfect divergence.
    X['flow_sync'] = X['mean_momentum'] * np.sign(X['mean_delta'])

    # 2. Normalized Flow Magnitude: A composite of OBI and Delta intensity.
    # We normalize delta to [0, 1] to prevent scale issues, then multiply by OBI.
    # Result is in [0, 1].
    delta_norm = X['mean_delta'].abs() / (X['mean_delta'].abs() + 1.0)
    X['flow_mag'] = delta_norm * X['mean_obi'].abs()

    # 3. Exchange Agreement: Inverse of the price spread.
    # High spread = low agreement. Result is in (0, 1].
    X['agreement'] = 1.0 / (1.0 + X['price_spread'])

    # Drop metadata and target columns
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup: handle NaNs and infinities
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return XGBoost hyperparameters optimized for high regularization."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 5.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using Synchronized Flow-Agreement Gating.
    Multiplies the model's EV by a conviction score based on microstructure quality.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered conviction features
    # flow_sync: [-1, 1] (Alignment)
    # flow_mag: [0, 1] (Intensity)
    # agreement: (0, 1] (Consensus)
    sync = features_row.get('flow_sync', 0.0)
    mag = features_row.get('flow_mag', 0.0)
    agree = features_row.get('agreement', 1.0)

    # 3. Calculate Conviction Multiplier
    # We boost the EV if:
    # - Momentum and Delta are aligned (1 + sync)
    # - Flow magnitude is high (1 + mag)
    # - Exchanges agree (agree)
    # If sync is -1 (divergence), multiplier becomes 0, killing the trade.
    multiplier = (1.0 + sync) * (1.0 + mag) * agree
    effective_ev = best_ev * multiplier

    # 4. Execution Logic
    # Base hurdle for effective edge
    hurdle = 0.03

    if effective_ev > hurdle:
        # Time-decay: Reduce exposure as the window closes to mitigate last-second noise
        # Scale from 1.0 (300s) down to 0.2 (0s)
        time_factor = max(0.2, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale based on how much the effective edge exceeds the hurdle
        # We use a 0.5 scaling coefficient to keep bets conservative
        bet_fraction = min(1.0, (effective_ev / hurdle) * 0.5) * time_factor

        # Minimum bet threshold to avoid dust
        if bet_fraction < 0.02:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0