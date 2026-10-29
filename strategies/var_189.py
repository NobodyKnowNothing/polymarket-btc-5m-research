# Var 189 | BT: 0.735 | Live PnL: $46.28 | Trades: 113
# Desc: Implements "Tri-Factor Congruence Scaling (TFCS)," which identifies high-conviction trades by detecting alignment (or calculated divergence) between three distinct market layers: Macro (Funding), Micr
# Stats: {"live_returns": [-0.7, -0.7, 0.9895704999999999, -0.5, -0.7, 0.05738350000000003, -0.7, -0.5, -0.7, -0.7, 1.3235936, -0.7, -0.7, -0.5, -0.5, 0.05738350000000003, 0.8637314, 1.6767869999999998, 0.08033690000000004, -0.5, -0.5, -0.7, 0.12384259999999991, 1.1587912, 0.9377459000000001, 1.0193575, 0.04591299999999998, -0.5, 0.014352999999999949, 0.08845899999999995, -0.5, 0.024428000000000005, -0.5, -0.7, -0.7, -0.7, 0.019337500000000007, -0.5, 0.06427819999999995, 0.06331500000000001, -0.7, 0.945424, 0.05738350000000003, 0.03493849999999998, -0.5, 0.019337500000000007, 0.014352999999999949, 1.0193575, -0.5, 0.142737, 0.07559349999999998, 46.1731385, -0.7, -0.5, -0.7, -0.5, 0.616951, -0.5, 0.8292494000000001, -0.5, -0.5, 0.02962699999999996, -0.7, -0.5, 0.02962699999999996, -0.7, -0.7, 1.6767869999999998, 2.7811874999999997, 0.9377459000000001, -0.5, 1.5964010999999998, -0.7, -0.7, -0.7, 1.9565461999999998, -0.7, -0.5, -0.7, 0.05651240000000002, -0.7, -0.7, -0.5, -0.5, 0.019337500000000007, 0.07221759999999999, 1.3235936, -0.7, -0.5, -0.5, 1.9565461999999998, -0.7, -0.5, 2.7811874999999997, -0.7, -0.5, -0.7, -0.7, 1.086697, 0.8637314, -0.5, 0.07221759999999999, -0.5, -0.5, -0.5, 1.9565461999999998, 0.06938299999999997, 3.1854522000000003, 1.6767869999999998, 0.7281124999999999, -0.5, 0.08864099999999997, -0.7], "live_pnl": 46.281955099999955, "live_wins": 51, "live_total": 113, "age": 862, "rounds_underwater": 0, "abstain_rounds": 15, "backtest_score": 0.734704937085968, "backtest_acc": 0.5168519182502689, "backtest_n": 702}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    congruence between funding, aggressive flow, and order book intent.
    """
    df = df.copy()

    # 1. Directional Sign Extraction
    # We use the sign of the three key drivers to calculate a 'congruence score'
    df['funding_sign'] = np.sign(df['mean_funding'])
    df['delta_sign'] = np.sign(df['mean_delta'])
    df['obi_sign'] = np.sign(df['mean_obi'])

    # 2. Congruence Score
    # Range: [-3, 3]. 
    # 3: All bullish (Funding, Delta, OBI). 
    # -3: All bearish.
    # -1: Funding is positive (longs), but Delta/OBI are negative (selling) -> Long Squeeze.
    # 1: Funding is negative (shorts), but Delta/OBI are positive (buying) -> Short Squeeze.
    df['congruence_score'] = df['funding_sign'] + df['delta_sign'] + df['obi_sign']

    # 3. Magnitude features for scaling
    df['funding_mag'] = df['mean_funding'].abs()
    df['delta_mag'] = df['mean_delta'].abs()
    df['obi_mag'] = df['mean_obi'].abs()

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
        'n_estimators': 1500,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Factor Congruence Scaling (TFCS)".

    Logic:
    1. Trend Following (High Congruence):
       If |congruence| >= 2, the three factors (Funding, Delta, OBI) are in strong alignment.
       We trade the direction of the model if the EV is sufficient.
    
    2. Liquidation Squeezes (Divergent Congruence):
       - Long Squeeze: Funding is positive (longs paying), but Delta/OBI are negative.
         This suggests aggressive selling into long positions.
       - Short Squeeze: Funding is negative (shorts paying), but Delta/OBI are positive.
         This suggests aggressive buying into short positions.

    3. Execution:
       - Trend trades: Moderate bet size, requires standard EV hurdle.
       - Squeeze trades: Larger bet size, targets high-velocity moves.
    """
    # Exit if too little time remains
    if seconds_remaining < 40:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    congruence = features_row.get('congruence_score', 0.0)
    funding_sign = features_row.get('funding_sign', 0.0)
    
    side = None
    bet_fraction = 0.0

    # --- REGIME 1: TREND FOLLOWING (|Congruence| >= 2) ---
    # All three factors (or at least two) are pointing in the same direction.
    if abs(congruence) >= 2:
        if congruence > 0 and ev_up > 0.025:
            side = 'Up'
            bet_fraction = 0.5
        elif congruence < 0 and ev_down > 0.025:
            side = 'Down'
            bet_fraction = 0.5

    # --- REGIME 2: LIQUIDATION SQUEEZES (Specific Divergence) ---
    # A squeeze occurs when the 'macro' (funding) is at odds with the 'micro' (flow/intent).
    
    # Long Squeeze: Funding is (+) but flow/intent is (-)
    elif congruence == -1 and funding_sign > 0:
        if ev_down > 0.02:
            side = 'Down'
            bet_fraction = 0.7  # Squeezes are high-conviction/high-velocity

    # Short Squeeze: Funding is (-) but flow/intent is (+)
    elif congruence == 1 and funding_sign < 0:
        if ev_up > 0.02:
            side = 'Up'
            bet_fraction = 0.7

    # Final safety clamp
    if side:
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0