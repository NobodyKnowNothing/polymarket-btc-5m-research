# Var 202 | BT: -1.020 | Live PnL: $57.61 | Trades: 102
# Desc: variant 1
# Stats: {"live_returns": [-0.6320662562678434, 2.2313163824924462, -0.9838391312782896, 3.1176451906208316, -0.47735029092633896, -0.8611555637886599, -1.0, -0.4995916544168271, 0.18908480000000003, -0.8683416294083025, -0.404861024610319, -0.4850722591749179, -0.42963527826092435, 1.3866454027789616, -0.4995237486948792, -0.8284891567213942, -0.27674532624850484, -0.9428813864024955, -0.21731839281497145, -0.804662296837545, -0.9981893240752374, -0.2941458661384806, -0.9300485050563881, -0.21790625120623447, 1.2443685525712818, -0.1, -0.23625610448568987, -0.12823992738156909, 1.1369867808946112, -0.9405215652002017, -0.05, -0.4999996719957038, -0.49999997229606363, 65.62605190890432, -0.21241574882697484, -0.4680338829778298, -0.8098005856523416, -0.4885572850255218, -0.499999999994993, -0.108735434753748, -0.2736959267956779, -0.3966087189316961, -0.4428160860485137, -0.49932948674375544, -0.19474546866508674, 1.2296655401075098, 2.305895774738208, -0.29434197064821915, 1.0179002306226788, -0.43171267551050985, -0.5459545009683583, -0.2747087575107621, -0.49998974048013317, -0.4998788410985536, -0.9399213046106827, -0.9992517871417105, 2.1208992649097134, -0.49578084280252566, -0.4998216783941878, -0.49427856303899936, -0.4928217643576745, -0.08896767996232544, 1.3433365212218327, 3.90277615063619, -0.4912736759765114, 2.275322836763861, 1.8346250462162093, -0.4351238350283457, -0.9144838792636907, 1.3154559162970514, -0.5428937342723453, -0.45110957810495544, -0.9635868098583399, -0.7319195694783304, -0.9848575078121204, -0.9999786542471639, 0.8899643537853004, -0.1, -0.5612455744455799, -0.9772566604449047, -0.23569258821935044, 1.0854468373880237, -0.4247006628895822, -0.4966381479343943, 0.562086853749191, -0.48385699671069626, -0.4872197731022372, -0.490243267421313, -0.6749063344053693, -0.4999999986312799, -0.9999914548596742, -0.920573542576715, -0.4494633700945696, 0.27950660000000005, -0.9957713239159275, 0.8090743144936157, 1.3829765743734068, 1.6535151143986264, 1.728634356997493, -0.9974925875951316, -0.4288329328343066, -0.26482181249897585], "live_pnl": 57.60623771771102, "live_wins": 24, "live_total": 102, "age": 799, "rounds_underwater": 0, "abstain_rounds": 0, "backtest_score": -1.0195470943485974, "backtest_acc": 0.5586463501063076, "backtest_n": 346}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of aggressive flow and the dispersion of exchange prices.
    """
    df = df.copy()

    # 1. Flow Efficiency: How much aggressive flow (sum_delta) is moving the price 
    # relative to the current volatility regime. We use tanh to squash the ratio.
    # High efficiency = momentum/vacuum; Low efficiency = absorption.
    df['flow_efficiency'] = np.tanh(df['sum_delta'] / (df['mean_volatility'] + 1e-6))
    
    # 2. Dispersion Index: Measures how much the exchanges disagree (price_spread)
    # relative to the current volatility. High dispersion suggests a fragmented/unstable market.
    df['dispersion_index'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 3. Momentum-Volatility Ratio: Normalizes momentum by the current volatility regime.
    df['norm_momentum'] = df['mean_momentum'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on the most robust flow signals."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.015,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Volatility-Normalized Flow-Price Coherence (VNFPC)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Dispersion Gating: Increase the EV hurdle if the 'dispersion_index' is high, 
       meaning exchanges are in disagreement.
    3. Efficiency Scaling: Scale the bet size by the absolute 'flow_efficiency'. 
       We want to bet most when aggressive flow is efficiently moving price (momentum)
       and least when flow is being absorbed (low efficiency).
    4. Time Decay: Reduce conviction as the window closes to avoid late-stage noise.
    """
    # Exit if too little time remains
    if seconds_remaining < 20:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # Extract engineered features
    flow_eff = features_row.get('flow_efficiency', 0.0)
    dispersion = features_row.get('dispersion_index', 0.0)

    # --- 1. DYNAMIC HURDLE CALCULATION ---
    # Base hurdle is 0.03. 
    # Increase hurdle if exchange prices are diverging (high dispersion).
    # Increase hurdle as time runs out to avoid volatility spikes at close.
    time_factor = (seconds_remaining / 300.0) # 1.0 at start, ~0.06 at 280s remaining
    hurdle = 0.03 + (dispersion * 0.02) + (0.02 * (1.0 - time_factor))

    # --- 2. EXECUTION ---
    if best_ev > hurdle:
        # Bet size is scaled by the magnitude of flow efficiency.
        # If flow_eff is near 0 (absorption), bet_fraction is near 0.
        # If flow_eff is near 1 (clean momentum), bet_fraction is near 1.
        bet_fraction = abs(flow_eff)

        # Cap the bet fraction to a reasonable maximum
        bet_fraction = min(1.0, max(0.1, bet_fraction))
        
        # Apply an additional penalty if dispersion is extreme (> 5.0)
        if dispersion > 5.0:
            bet_fraction *= 0.5

        return side, bet_fraction

    return None, 0.0