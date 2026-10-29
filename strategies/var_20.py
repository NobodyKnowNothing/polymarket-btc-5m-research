# Var 20 | BT: 3.028 | Live PnL: $0.52 | Trades: 20
# Desc: Implements "Cross-Exchange Delta-OBI Congruence & Volatility-Adjusted Edge," which requires directional agreement of the Delta-OBI product across exchanges and scales the entry hurdle by the current v

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the consensus of 'Aggression-Liquidity' (Delta * OBI) across exchanges.
    """
    # Drop metadata but keep pm_costs and seconds_remaining
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Calculate Delta-OBI products for each exchange
    # This represents the 'impact potential' of the current flow
    X['binance_delta_obi'] = X['binance_delta'] * X['binance_obi']
    X['coinbase_delta_obi'] = X['coinbase_delta'] * X['coinbase_obi']
    X['kraken_delta_obi'] = X['kraken_delta'] * X['kraken_obi']

    # 2. Consensus Magnitude: Average intensity of the Delta-OBI signal
    X['consensus_mag'] = (abs(X['binance_delta_obi']) + abs(X['coinbase_delta_obi']) + abs(X['kraken_delta_obi'])) / 3.0

    # 3. Consensus Sign: Directional agreement of the Delta-OBI product
    X['consensus_sign'] = np.sign(X['binance_delta_obi'] + X['coinbase_delta_obi'] + X['kraken_delta_obi'])

    # 4. Consensus Score: How many exchanges agree on the direction of Delta-OBI?
    # 2 = Full agreement, 1 = Partial, 0 = Disagreement/Noise
    X['sign_b'] = np.sign(X['binance_delta_obi'])
    X['sign_c'] = np.sign(X['coinbase_delta_obi'])
    X['sign_k'] = np.sign(X['kraken_delta_obi'])
    
    # Agreement only counts if the signal is non-zero
    mask_bc = (X['sign_b'] == X['sign_c']) & (X['sign_b'] != 0)
    mask_ck = (X['sign_c'] == X['sign_k']) & (X['sign_c'] != 0)
    X['agreement_score'] = mask_bc.astype(int) + mask_ck.astype(int)

    # 5. Volatility-Normalized Flow Intensity
    X['vol_delta_ratio'] = X['mean_delta'] / (X['mean_volatility'] + 1e-7)

    # Cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to handle interaction noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 40.0,
        'reg_lambda': 60.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade based on Cross-Exchange Congruence.
    
    We scale the required Expected Value (EV) hurdle based on:
    1. Volatility (higher vol -> higher hurdle)
    2. Consensus (lower agreement -> higher hurdle)
    3. Time Decay (approaching close -> higher hurdle to avoid noise)
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered signals
    consensus_sign = features_row.get('consensus_sign', 0)
    agreement_score = features_row.get('agreement_score', 0)
    consensus_mag = features_row.get('consensus_mag', 0)
    mean_vol = features_row.get('mean_volatility', 0.01)

    # 1. Base Hurdle
    base_hurdle = 0.025

    # 2. Volatility Adjustment: Scale hurdle up during high volatility regimes
    vol_adj = 1.0 + (mean_vol * 15.0)

    # 3. Agreement Adjustment: If agreement_score is 0, hurdle is much higher
    # score 2 -> 1.0x, score 1 -> 1.25x, score 0 -> 1.5x
    agreement_adj = 1.5 - (agreement_score * 0.25)

    # 4. Time Decay: Increase hurdle as the window closes to avoid late-stage noise
    # As seconds_remaining -> 0, time_adj -> 2.0
    time_adj = 1.0 + (1.0 - (min(300.0, seconds_remaining) / 300.0))

    total_hurdle = base_hurdle * vol_adj * agreement_adj * time_adj

    side = None
    best_ev = 0.0

    # Decision Logic: Match EV direction with the consensus direction
    # If consensus_sign is 0, we require a much higher EV to overcome uncertainty
    if ev_up > total_hurdle and consensus_sign >= 0:
        side = 'Up'
        best_ev = ev_up
    elif ev_down > total_hurdle and consensus_sign <= 0:
        side = 'Down'
        best_ev = ev_down
    # Emergency exit: if EV is massive, ignore consensus (rare edge)
    elif ev_up > (total_hurdle * 2.5):
        side = 'Up'
        best_ev = ev_up
    elif ev_down > (total_hurdle * 2.5):
        side = 'Down'
        best_ev = ev_down

    if side is None:
        return None, 0.0

    # DYNAMIC SIZING
    # Scale bet by the ratio of edge to hurdle, capped at 1.0
    # We also damp the size if the consensus magnitude is low
    conviction = (best_ev / total_hurdle)
    mag_factor = min(1.0, consensus_mag * 10.0) # Assuming consensus_mag is roughly scaled
    
    # If consensus is low, we still bet, but with much less conviction
    bet_fraction = min(1.0, conviction * 0.4 * (0.5 + 0.5 * mag_factor))

    # Threshold to ensure meaningful trades
    if bet_fraction < 0.1:
        return None, 0.0

    return side, bet_fraction