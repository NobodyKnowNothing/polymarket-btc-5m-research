# Var 184 | BT: 0.840 | Live PnL: $-3.68 | Trades: 60
# Desc: Implements "Intent-Action Velocity Divergence (IAVD)," which differentiates between "Coiling" regimes (where order book intent/OBI is growing faster than aggressive flow/Delta) and "Breakout" regimes
# Stats: {"live_returns": [0.06917002254543225, 0.03813958273065021, 0.04815479777734133, 0.045906800000000025, 0.09032850225813749, 0.03793664527336171, 0.12402029228354261, 0.058296676241507694, 0.12509815109962819, 0.10208218975011696, -0.9988267184857, 1.2322624395983097, 0.06843662395351224, -0.9989109220967974, 0.03801487152573524, 0.03229280000000001, -0.992437745427233, 0.11313121446202179, -0.4, 0.05065200000000003, -0.9920360219370634, -0.9950969307498264, 0.047577458581790855, -0.4, 0.03819870565382, 0.09091811632109636, 0.11021604796627393, 0.036730399999999996, 0.06555999999999995, 0.06805224361284934, 1.8844033784198244, 0.06889168803494017, -0.9983138666016823, 0.03760054723229911, 0.037932495372362096, -0.9916229623417226, -0.9993513203078245, 0.1495955354232692, 0.028053587413293446, 0.03229280000000001, -0.4, 0.060474799999999995, 0.12472660879582809, -0.9933765601403705, 0.0480463631879442, 1.5854279999999998, -0.9972890949437836, -0.9849761166447721, 0.028102845117734976, 0.036730399999999996, 0.03789306655756264, 0.03762644005441795, -0.9993871769193131, 0.019542400000000015, -0.9987662074821037, 0.21421226145522487, 1.0905625241354602, 0.02370159999999999, 2.058943589044712, 0.12472014694973721], "live_pnl": -3.6797339852484545, "live_wins": 44, "live_total": 60, "age": 758, "rounds_underwater": 44, "abstain_rounds": 0, "backtest_score": 0.8401967904722951, "backtest_acc": 0.5502778275676644, "backtest_n": 196}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    velocity divergence between order book intent (OBI) and aggressive flow (Delta).
    """
    df = df.copy()

    # 1. Velocity of Intent (OBI) and Action (Delta)
    # We group by market_slug to ensure diffs are calculated within the same 5-min window.
    df['obi_vel'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Intent-Action Gap (IAG)
    # Positive IAG: Intent (OBI) is increasing faster than Action (Delta) -> "Coiling"
    # Negative IAG: Action (Delta) is increasing faster than Intent (OBI) -> "Breakout"
    df['iag'] = df['obi_vel'] - df['delta_vel']
    df['iag_mag'] = df['iag'].abs()

    # 3. Price-Flow Efficiency (How much delta is required to move price)
    # We add a small epsilon to avoid division by zero.
    df['price_move_per_delta'] = df['mean_delta'].abs() / (df['price_vs_strike_pct'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Intent-Action Velocity Divergence (IAVD)".

    Logic:
    1. Identify Regime:
       - Coiling Regime (iag > threshold): Intent is leading. We expect a breakout soon.
         Bet smaller, higher hurdle, early in the window.
       - Breakout Regime (iag < -threshold): Action is leading. The move is happening.
         Bet larger, lower hurdle, mid-window.
    2. Execution:
       - In Coiling: We are 'predicting' the move. We require higher model confidence.
       - In Breakout: We are 'riding' the move. We require lower model confidence but 
         scale down if the price is already significantly extended from the strike.
    """
    # Exit if too little time remains to capture alpha
    if seconds_remaining < 45:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    iag = features_row.get('iag', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    price_dist = features_row.get('price_vs_strike_pct', 0.0)
    
    side = None
    bet_fraction = 0.0
    
    # Threshold for regime detection
    threshold = 0.005 

    # --- REGIME 1: COILING (Intent Leading) ---
    if iag > threshold:
        # We expect a move to follow the direction of the growing intent/flow
        if mean_delta > 0 and prob_up > 0.53 and ev_up > 0.04:
            side = 'Up'
            # Smaller bet: we are entering before the momentum is confirmed
            bet_fraction = 0.4 
        elif mean_delta < 0 and prob_down > 0.53 and ev_down > 0.04:
            side = 'Down'
            bet_fraction = 0.4
            
    # --- REGIME 2: BREAKOUT (Action Leading) ---
    elif iag < -threshold:
        # We ride the existing aggressive flow
        if mean_delta > 0 and prob_up > 0.51 and ev_up > 0.02:
            side = 'Up'
            # Larger bet: the move is in progress, but scale down if price is extended
            # Avoid buying the "top" of a 5-min candle
            extension_penalty = 1.0 / (1.0 + abs(price_dist) * 20)
            bet_fraction = 1.0 * extension_penalty
        elif mean_delta < 0 and prob_down > 0.51 and ev_down > 0.02:
            side = 'Down'
            extension_penalty = 1.0 / (1.0 + abs(price_dist) * 20)
            bet_fraction = 1.0 * extension_penalty

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.05, bet_fraction))
    
    return None, 0.0