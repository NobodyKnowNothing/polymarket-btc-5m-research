# Var 83 | BT: 0.468 | Live PnL: $-0.13 | Trades: 98
# Desc: Implements "Microstructure Regime-Switching (MRS)," which uses the alignment (or divergence) between Order Book Imbalance (OBI) and Signed Delta to classify the market into "Impulse" or "Absorption" r
# Stats: {"live_returns": [-0.05, 0.0055010318756103516, 0.17664332687854767, -0.3697505295276642, 2.0265421867370605, 0.12370090186595917, -0.3739077150821686, -0.05173315480351448, -0.10637979954481125, -0.7833344340324402, -0.15865135192871094, 0.7814890146255493, 0.007877267897129059, -0.1823718249797821, -0.23575887084007263, -0.12246447056531906, 0.006331500000000004, 0.41062304377555847, -0.46125784516334534, -0.3779451847076416, -0.43224698305130005, -0.22149723768234253, 0.872994601726532, -0.49499091506004333, 0.22753230000000002, -0.4317891597747803, -0.17004527151584625, -0.7347166538238525, 1.1496752500534058, 0.3658663034439087, 0.35783150792121887, 1.7777453660964966, -0.14051610231399536, -0.06264153122901917, 3.303756, -0.14841115474700928, -0.5305006504058838, 1.329910159111023, -1.0, 0.008664846420288086, -0.11191185563802719, 0.011680260300636292, -0.6663112640380859, -0.5986849069595337, -0.12777957320213318, -0.11566895991563797, 0.9060016870498657, 0.97907954454422, 0.36250385642051697, -0.17284153401851654, -0.09484435617923737, -0.09298352152109146, -0.4539988338947296, 0.4711328446865082, 0.18908295035362244, 0.005274534225463867, 0.018447235226631165, -1.0, 0.014268971979618073, -0.05, -0.05310387909412384, 0.24690604209899902, -0.33162593841552734, 0.0206223726272583, -0.2155664712190628, 0.3612179160118103, -0.3040387034416199, 0.21470753848552704, 0.780805766582489, -0.5495255589485168, -0.3710881769657135, -0.2803525924682617, -0.1473412662744522, 0.00993049144744873, -0.8729033470153809, -0.094390369951725, 0.45125964283943176, 0.33096009492874146, 0.01450326293706894, -0.7974259853363037, 0.0175180584192276, 0.780274510383606, 0.7295308113098145, -0.3442866802215576, -0.99429851770401, 0.27932125329971313, -0.23284535109996796, -0.22581492364406586, -0.4317905604839325, 0.11753980070352554, 0.0189468115568161, -0.43168318271636963, -0.07466582953929901, -0.4147317111492157, -0.4467381238937378, -1.0, -0.19679483771324158, -0.4548313617706299], "live_pnl": -0.1275777518749237, "live_wins": 41, "live_total": 98, "age": 690, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.4683143167738386, "backtest_acc": 0.5230271346776201, "backtest_n": 846}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between order book intent (OBI) and actual flow (Delta).
    """
    df = df.copy()

    # 1. Pressure Index: Product of OBI and Delta.
    # Positive = Impulse (OBI and Delta agree).
    # Negative = Absorption (OBI and Delta disagree).
    df['pressure_index'] = df['mean_obi'] * df['mean_delta']

    # 2. Spread Uncertainty: Combined measure of price and book dispersion.
    # Higher values indicate a noisier, less reliable microstructure.
    df['spread_uncertainty'] = df['price_spread'] + df['obi_spread']

    # 3. Price-to-Strike Velocity: How fast the price is moving relative to the strike.
    # We use the percentage distance to normalize across different BTC price levels.
    df['price_velocity'] = df['price_vs_strike_pct'] / (df['seconds_remaining'] + 1e-6)

    # 4. Delta Intensity: Magnitude of the aggressive flow.
    df['delta_intensity'] = df['mean_delta'].abs()

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
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Regime-Switching (MRS)".

    Logic:
    1. Identify Regime: 
       - Impulse (pressure > 0): Flow and Intent align. We trade with the flow.
       - Absorption (pressure < 0): Flow and Intent conflict. We trade against the flow.
       - Neutral (pressure ~ 0): We abstain to protect Sharpe.
    2. Directional Validation:
       - In Impulse: If delta is +, we look for 'Up' trades. If delta is -, we look for 'Down'.
       - In Absorption: If delta is +, we look for 'Down' trades (reversal). If delta is -, we look for 'Up'.
    3. Uncertainty Scaling:
       - Scale bet size inversely with 'spread_uncertainty'.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    pressure = features_row.get('pressure_index', 0.0)
    uncertainty = features_row.get('spread_uncertainty', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    delta_sign = np.sign(mean_delta)

    side = None
    best_ev = 0.0

    # Threshold for regime classification
    regime_threshold = 0.05 

    # 1. Regime-Based Decision Logic
    if pressure > regime_threshold:
        # --- IMPULSE REGIME ---
        # We follow the direction of the delta
        if delta_sign > 0 and ev_up > 0.03:
            side = 'Up'
            best_ev = ev_up
        elif delta_sign < 0 and ev_down > 0.03:
            side = 'Down'
            best_ev = ev_down

    elif pressure < -regime_threshold:
        # --- ABSORPTION REGIME ---
        # We trade the reversal (against the delta)
        if delta_sign > 0 and ev_down > 0.03:
            side = 'Down'
            best_ev = ev_down
        elif delta_sign < 0 and ev_up > 0.03:
            side = 'Up'
            best_ev = ev_up

    # 2. Final Execution
    if side:
        # Scale bet fraction by:
        # a) The magnitude of the edge (target 4% EV for full bet)
        # b) The inverse of market uncertainty (spreads)
        # c) Time decay (increase caution as time runs out)
        
        time_factor = 1.0 if seconds_remaining > 60 else (seconds_remaining / 60.0)
        uncertainty_penalty = 1.0 / (1.0 + uncertainty)
        
        bet_fraction = (best_ev / 0.04) * uncertainty_penalty * time_factor
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0