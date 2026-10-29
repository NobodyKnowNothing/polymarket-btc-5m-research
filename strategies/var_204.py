# Var 204 | BT: 0.270 | Live PnL: $-3.84 | Trades: 60
# Desc: Implements "Consensus-Gated Alignment & Absorption (CGAA)," which uses the granular agreement level across exchanges to dynamically scale the required edge (hurdle) and the bet size, while adjusting f
# Stats: {"live_returns": [-0.33270783357924266, 1.9293298690353913, 0.754462810908241, -0.3324772122912185, -0.3328150839783001, 2.0642505832414213, -0.3253298945643335, 0.6567656118405261, 0.12338458647250039, 0.07980966758829322, 1.0042901402932394, -0.3323241002862568, 0.6181453391694497, -0.99730733091153, -0.9934250154830285, 0.8797299264280143, -0.6656892187768018, -0.33253818113895484, -0.33097435386455654, -0.3308915075569383, -0.32971852076035535, -0.6606123236208916, -0.3301883397385621, -0.6662541899812298, -0.9990236058758938, -0.6637217428000681, -0.6635028252877524, 0.7189122729260687, -0.3314228449079749, -0.33214565884897784, -0.33049635563041885, 0.12459499491538262, -0.9807102161732578, 0.5492231289464828, 0.883616490657209, -0.3309698226133514, 1.6718779222141928, -0.33182072306142496, -0.33328299718625487, -0.31646944562858026, 0.9675513716062889, -0.3319107134173903, 0.11773803556274076, -0.32880609312633524, -0.33034129318131067, -0.3306346951559228, 0.6941656943872316, -0.32550033106236786, 0.5466901683768821, -0.6585182580712363, -0.3300216027778167, 1.2385749476663261, -0.3325315389976758, -0.3295277698985996, -0.6610327355045751, -0.9973750168307763, 0.028517809887545376, -0.3298273815379483, -0.3121978582632505, -0.9874223301574161], "live_pnl": -3.8408355904053493, "live_wins": 20, "live_total": 60, "age": 163, "rounds_underwater": 8, "abstain_rounds": 0, "backtest_score": 0.2702104752641728, "backtest_acc": 0.563016347336966, "backtest_n": 2937}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange consensus and the alignment between intent (OBI) and action (Delta).
    """
    df = df.copy()

    # 1. Velocity of Delta (to capture acceleration)
    df['delta_vel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Cross-Exchange Consensus Scores
    # Sum of signs: 3 = total agreement, 1 = majority, 0 = chaos/neutral, -1 = majority, -3 = total disagreement
    df['delta_consensus'] = (np.sign(df['binance_delta']) + 
                             np.sign(df['coinbase_delta']) + 
                             np.sign(df['kraken_delta']))
    
    df['obi_consensus'] = (np.sign(df['binance_obi']) + 
                           np.sign(df['coinbase_obi']) + 
                           np.sign(df['kraken_obi']))

    # 3. Delta-OBI Alignment (Trend vs. Absorption)
    # 1: Delta and OBI agree (Trend/Impact)
    # -1: Delta and OBI disagree (Absorption/Friction)
    # 0: One or both are neutral
    df['delta_obi_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.009,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Consensus-Gated Alignment & Absorption (CGAA)".

    Logic:
    1. Consensus Check: Calculate how many exchanges agree on the Delta sign.
       - If consensus is low (e.g., only 1 exchange agrees), significantly increase the EV hurdle.
    2. Alignment Check:
       - Trend (Alignment=1): Delta and OBI move together. Lower hurdle, higher confidence.
       - Absorption (Alignment=-1): Delta moves against OBI. Higher hurdle (risk of reversal).
    3. Dynamic Sizing:
       - Scale bet size by consensus magnitude and penalize for price extension from strike.
    """
    # Exit if too little time remains
    if seconds_remaining < 35:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Extract engineered features
    delta_consensus_mag = abs(features_row.get('delta_consensus', 0.0))
    alignment = features_row.get('delta_obi_alignment', 0.0)
    price_dist = abs(features_row.get('price_vs_strike_pct', 0.0))

    # If there is no directional consensus among exchanges, abstain
    if delta_consensus_mag < 1.0:
        return None, 0.0

    # --- 1. Determine the Hurdle (Required EV) ---
    # Base hurdle depends on the regime (Trend vs Absorption)
    if alignment > 0:
        base_hurdle = 0.020  # Trend: easier to ride
    elif alignment < 0:
        base_hurdle = 0.040  # Absorption: harder to trade, higher risk of whip
    else:
        base_hurdle = 0.030  # Neutral

    # Scale hurdle by consensus: if consensus is low (mag=1), hurdle is much higher
    # mag=3 -> scale=1.0; mag=1 -> scale=3.0
    consensus_hurdle_scale = 3.0 / max(delta_consensus_mag, 1.0)
    final_hurdle = base_hurdle * consensus_hurdle_scale

    # --- 2. Execution Logic ---
    side = None
    if ev_up > final_hurdle and ev_up > ev_down:
        side = 'Up'
        current_ev = ev_up
    elif ev_down > final_hurdle and ev_down > ev_up:
        side = 'Down'
        current_ev = ev_down
    else:
        return None, 0.0

    # --- 3. Dynamic Bet Sizing ---
    # Scale by consensus (mag=3 -> 1.0, mag=1 -> 0.33)
    consensus_scale = delta_consensus_mag / 3.0
    
    # Scale by price extension (avoid buying the top/bottom of a move)
    # As price_dist increases, extension_penalty decreases
    extension_penalty = 1.0 / (1.0 + price_dist * 25.0)
    
    bet_fraction = consensus_scale * extension_penalty

    # Final safety clamp
    return side, min(1.0, max(0.05, bet_fraction))