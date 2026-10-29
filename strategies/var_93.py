# Var 93 | BT: -0.483 | Live PnL: $-11.36 | Trades: 60
# Desc: Implements "Squeeze-Model Synergy (SMS)," which uses the interaction between perp funding and aggressive delta to identify squeeze regimes, then uses the ML model to either amplify the squeeze convict
# Stats: {"live_returns": [-0.906818687915802, 0.042206764221191406, -1.0, -1.0, -1.0, 1.28551, -1.0, -0.7361631393432617, -1.0, -1.0, 3.9731249999999996, -0.41160109639167786, 2.173394, -1.0, -1.0, -1.0, -1.0, -1.0, -0.3695860207080841, -1.0, -1.0, 1.4219911098480225, -1.0, 2.39541, -1.0, -0.40726280212402344, 0.054585814476013184, 1.1385122537612915, -1.0, 1.637774109840393, 3.1194439999999997, -1.0, -1.0, -1.0, -0.6507341265678406, -0.5763207674026489, -1.0, 2.39541, 1.0193400382995605, -0.9141308665275574, -1.0, -1.0, -1.0, 1.9791409999999998, -0.3818587064743042, -0.3930750787258148, -0.392226904630661, -1.0, 0.09055280685424805, -1.0, -1.0, 0.033822596073150635, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -0.5152856111526489, 1.7294070000000001, -1.0], "live_pnl": -11.357684135437012, "live_wins": 17, "live_total": 60, "age": 126, "rounds_underwater": 60, "abstain_rounds": 1, "backtest_score": -0.4826817556257383, "backtest_acc": 0.561152821506418, "backtest_n": 2586}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    squeeze intensity and cross-exchange consensus.
    """
    df = df.copy()

    # 1. Squeeze Intensity: Interaction between funding and aggressive flow.
    # High values indicate a potential squeeze (funding and delta in opposite directions).
    df['squeeze_intensity'] = df['mean_funding'].abs() * df['mean_delta'].abs()

    # 2. OBI-Delta Alignment: Do the order book and aggressive flow agree?
    df['obi_delta_alignment'] = df['mean_obi'] * np.sign(df['mean_delta'])

    # 3. Exchange Agreement: Sum of OBI signs across the three exchanges (-3 to 3).
    df['exchange_agreement'] = (
        np.sign(df['binance_obi']) + 
        np.sign(df['coinbase_obi']) + 
        np.sign(df['kraken_obi'])
    )

    # 4. Volatility-Normalized Delta: Flow relative to current volatility.
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters for robust microstructure signal detection."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Squeeze-Model Synergy (SMS)".

    Logic:
    - Identify "Squeeze Regimes": 
        - Long Squeeze: Positive funding + Aggressive selling (Delta < 0).
        - Short Squeeze: Negative funding + Aggressive buying (Delta > 0).
    - Synergy Check:
        - If the model's predicted side matches the squeeze direction, amplify EV (synergy = 1.5).
        - If the model contradicts the squeeze, heavily penalize EV (synergy = 0.1).
    - Consensus Multiplier: If at least 2 exchanges show OBI agreement, boost conviction.
    - Time-Decay: Scale down bet size as the market window closes to avoid noise.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    mean_funding = features_row.get('mean_funding', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    agreement_score = features_row.get('exchange_agreement', 0.0)

    # 1. Identify Squeeze Direction
    squeeze_side = None
    # Long Squeeze: Longs paying to stay long, but aggressive selling is happening.
    if mean_funding > 0 and mean_delta < -0.005:
        squeeze_side = 'Down'
    # Short Squeeze: Shorts paying to stay short, but aggressive buying is happening.
    elif mean_funding < 0 and mean_delta > 0.005:
        squeeze_side = 'Up'

    # 2. Apply Synergy Multiplier
    synergy = 1.0
    if squeeze_side:
        if side == squeeze_side:
            synergy = 1.5
            # Boost if exchanges agree on the OBI direction
            if abs(agreement_score) >= 1:
                synergy *= 1.2
        else:
            # Model is fighting a clear squeeze regime
            synergy = 0.1

    # 3. Calculate Effective EV
    effective_ev = best_ev * synergy

    # 4. Execution Logic
    base_threshold = 0.03
    if effective_ev > base_threshold:
        # Target a 4% edge for a full 1.0 bet
        bet_fraction = effective_ev / 0.04
        
        # 5. Time-Decay: Reduce bet size as the window closes (300s window)
        # This prevents over-leveraging on high-noise, last-second predictions.
        time_decay = min(1.0, seconds_remaining / 300.0)
        
        return side, min(1.0, bet_fraction * time_decay)

    return None, 0.0