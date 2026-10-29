# Var 122 | BT: 0.122 | Live PnL: $-0.65 | Trades: 146
# Desc: Implements "Funding-Stress Squeeze Gating (FSSG)," which identifies high-conviction squeeze regimes by calculating the directional tension between perp funding rates and aggressive order flow, amplify
# Stats: {"live_returns": [-1.0, -1.0, 1.5854279999999998, 2.280573, 1.8077519999999998, -1.0, -1.0, -1.0, 4.893558, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, 2.651597, -1.0, 2.950496, -1.0, 1.7645650000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.11476700000000006, 1.890848, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, 4.550646, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 2.795066, 3.727685, 3.1194439999999997, -1.0, 1.9791409999999998, -1.0, -1.0, 3.303756, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, 2.280573, -1.0, -1.0, -1.0, -1.0, 1.6554160000000002, -1.0, -1.0, 2.073133, 2.280573, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, 1.5191249999999998, -1.0, 2.39541, 1.28551, 2.280573, 1.9791409999999998, -1.0, 1.890848, -1.0, -1.0, -1.0, 1.890848, 1.7294070000000001, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, 6.238865, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], "live_pnl": -0.6545789999999956, "live_wins": 43, "live_total": 146, "age": 346, "rounds_underwater": 1, "abstain_rounds": 1, "backtest_score": 0.12213482901500308, "backtest_acc": 0.5441613588110403, "backtest_n": 2952}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the tension between funding rates and aggressive order flow.
    """
    df = df.copy()

    # 1. Funding-Delta Tension
    # A high absolute value indicates a potential squeeze (e.g., high funding but negative delta)
    df['funding_delta_stress'] = df['mean_funding'] * df['mean_delta']

    # 2. Order Flow Congruence
    # Measures if OBI and Delta are moving in the same direction
    df['obi_delta_congruence'] = np.sign(df['mean_obi']) * np.sign(df['mean_delta'])

    # 3. Price-Flow Efficiency
    # How much price moves per unit of delta (detects absorption vs. vacuum)
    df['price_delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Stress Squeeze Gating (FSSG)".

    Logic:
    - We identify 'Squeeze Regimes' where funding and delta have opposite signs.
    - If funding is positive (longs paying) and delta is negative (selling), a 'Down Squeeze' is likely.
    - If funding is negative (shorts paying) and delta is positive (buying), an 'Up Squeeze' is likely.
    - We calculate a 'Squeeze Strength' based on the magnitude of this tension.
    - We amplify the model's EV if the prediction aligns with the squeeze direction.
    - We heavily penalize (or nullify) the EV if the prediction fights the squeeze.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 1. Calculate Squeeze Dynamics
    # squeeze_direction: 1 if Up Squeeze (funding < 0, delta > 0), -1 if Down Squeeze (funding > 0, delta < 0)
    # Note: We use the product of signs. If signs are opposite, product is -1.
    # We want to align 'side' with the direction of the squeeze.
    
    mf = features_row.get('mean_funding', 0.0)
    md = features_row.get('mean_delta', 0.0)
    
    # A squeeze exists when funding and delta signs are opposite
    # If mf > 0 and md < 0 -> Down Squeeze (direction -1)
    # If mf < 0 and md > 0 -> Up Squeeze (direction 1)
    squeeze_direction = 0
    if mf > 0 and md < 0:
        squeeze_direction = -1
    elif mf < 0 and md > 0:
        squeeze_direction = 1
    
    # Magnitude of the squeeze (scaled for sensitivity)
    # We use a multiplier to make the small funding rates impactful
    squeeze_strength = abs(mf * md) * 1000.0
    squeeze_strength = min(squeeze_strength, 2.0) # Cap the multiplier

    # 2. Apply Squeeze Gating to Expected Value
    effective_ev = best_ev
    
    if squeeze_direction != 0 and squeeze_strength > 0.1:
        # If the model side matches the squeeze direction, boost the conviction
        if (side == 'Up' and squeeze_direction == 1) or (side == 'Down' and squeeze_direction == -1):
            effective_ev = best_ev * (1.0 + squeeze_strength)
        else:
            # If the model fights the squeeze, penalize the EV significantly
            effective_ev = best_ev * (1.0 - squeeze_strength)

    # 3. Time-Decay Hurdle
    # As the window closes, we require a slightly higher effective EV to account for 
    # the diminishing opportunity to capture the move.
    time_hurdle = 0.02 + (10.0 / (seconds_remaining + 1.0))
    
    if effective_ev > time_hurdle:
        # 4. Dynamic Sizing
        # Scale bet size by the ratio of effective edge to a target 4% edge.
        # We also dampen the bet size by the squeeze strength if it's a 'fighting' trade.
        bet_fraction = (effective_ev / 0.04)
        
        # Ensure bet_fraction is sane
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0