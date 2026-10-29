# Var 44 | BT: -0.679 | Live PnL: $-1.47 | Trades: 61
# Desc: Implements "Multi-Exchange Microstructure Coherence (MEMC)," which shifts focus from single-exchange dominance to global consensus, requiring alignment between aggressive flow (Delta), passive book (O
# Stats: {"live_returns": [-0.15329940617084503, -0.4023068845272064, 0.2455359846353531, -0.27754324674606323, -0.35547545552253723, -0.2622230052947998, -0.10531850904226303, -0.43265366554260254, -0.11939779669046402, -0.3546753525733948, -0.07871010899543762, 0.008386246860027313, -0.11666393280029297, -0.09304416179656982, -0.5283222794532776, -0.33950865268707275, -0.15619970858097076, -0.4637184739112854, -0.8732207417488098, 0.004974029958248138, 0.11949416995048523, -0.13954252004623413, 0.6995412707328796, -0.512538492679596, 2.0078611373901367, -0.13559743762016296, -0.1273660957813263, -0.37125715613365173, -0.08324872702360153, -0.5213460326194763, 0.2329268604516983, -0.6152644157409668, 2.1304590702056885, -0.2256374955177307, 3.116011142730713, -0.18135897815227509, 0.008070744574069977, -0.35164159536361694, -0.2957272231578827, -0.22523704171180725, -0.08125574886798859, -0.058278560638427734, -0.30370795726776123, 0.7690789103507996, 0.004952386021614075, 0.1360848993062973, -0.12138635665178299, -0.22799067199230194, 0.18497340381145477, -0.15416982769966125, -0.3892006278038025, -0.12098664045333862, -0.21879415214061737, -0.13790713250637054, -0.2900024652481079, -0.4346965253353119, 0.5138030052185059, 0.02715305984020233, 0.0030151307582855225, -0.12702302634716034, -0.1142730563879013], "live_pnl": -1.4653959274291992, "live_wins": 17, "live_total": 61, "age": 71, "rounds_underwater": 14, "abstain_rounds": 0, "backtest_score": -0.6790694579131777, "backtest_acc": 0.5512915129151291, "backtest_n": 2357}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    global microstructure coherence and exchange consensus.
    """
    df = df.copy()

    # 1. Delta-OBI Sync: Do the aggregate aggressive flows align with the aggregate book pressure?
    # Positive if both are same sign, negative if they conflict.
    df['mean_delta_obi_sync'] = df['mean_delta'] * df['mean_obi']

    # 2. Delta Consensus: How much do the three exchanges agree on the direction of aggressive flow?
    # 1.0 = perfect agreement, < 0 = significant disagreement/conflict.
    delta_diff = (abs(df['binance_delta'] - df['coinbase_delta']) + 
                  abs(df['coinbase_delta'] - df['kraken_delta']) + 
                  abs(df['kraken_delta'] - df['binance_delta']))
    df['delta_agreement'] = 1.0 - (delta_diff / (df['mean_delta'].abs() + 1e-6))

    # 3. Funding-Flow Alignment: Is the aggressive flow moving with or against the perp funding?
    df['funding_flow_alignment'] = df['mean_funding'] * df['mean_delta']

    # 4. Price-to-Strike Momentum: How much has the price moved relative to the window start?
    # Helps the model understand if the current price is part of a sustained move.
    df['price_momentum_normalized'] = df['price_vs_strike_pct'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata but keep pm_costs and other features for the model
    # We must drop 'seconds_remaining' from X as it is metadata, not a feature.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return regularized XGBoost hyperparameters to capture subtle microstructure signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Multi-Exchange Microstructure Coherence (MEMC)".
    The strategy requires a 'confluence' of three signals: 
    1. Delta and OBI must agree on direction.
    2. All exchanges must show similar Delta (Consensus).
    3. The predicted edge must be sufficient to overcome the 'noise' of the regime.
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

    # 2. Extract Engineered Metrics
    sync = features_row.get('mean_delta_obi_sync', 0.0)
    agreement = features_row.get('delta_agreement', 0.0)
    mean_delta = features_row.get('mean_delta', 0.0)
    mean_obi = features_row.get('mean_obi', 0.0)
    
    # 3. Base Hurdle
    # We require a minimum edge to even consider a trade.
    base_hurdle = 0.025
    if best_ev < base_hurdle:
        return None, 0.0

    # 4. Confluence Multiplier
    # We scale our conviction based on how much the market agrees.
    multiplier = 1.0

    # A) Consensus Filter: If exchanges are fighting, reduce bet size significantly.
    # agreement is 1.0 for perfect consensus, < 0 for conflict.
    multiplier *= max(0.2, agreement)

    # B) Directional Congruence: Does the flow (Delta) match the book (OBI)?
    # If we are betting 'Up', we want both Delta and OBI to be positive.
    if side == 'Up':
        if mean_delta <= 0 or mean_obi <= 0:
            multiplier *= 0.4
    else: # side == 'Down'
        if mean_delta >= 0 or mean_obi >= 0:
            multiplier *= 0.4

    # 5. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid late-window volatility spikes.
    time_factor = (seconds_remaining / 300.0)
    
    # Final bet fraction: Scale by (Edge/Hurdle) and the confluence multiplier.
    # We cap the bet_fraction at 1.0.
    bet_fraction = (best_ev / base_hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure the trade is mathematically significant
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction