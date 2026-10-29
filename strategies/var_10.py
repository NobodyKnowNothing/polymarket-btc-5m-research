# Var 10 | BT: -0.546 | Live PnL: $-5.57 | Trades: 61
# Desc: Implements "Microstructure Efficiency & Consensus Gating," which uses a dynamic hurdle threshold that increases during periods of high absorption (low price-to-delta efficiency) or low exchange agreem
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.0261764526367188, 3.1194439999999997, 1.0784149169921875, -0.8410124778747559, -0.7864850759506226, -0.64182049036026, 1.890848, -0.9111728072166443, -0.77371746301651, -1.0, -0.7532265186309814, -0.5062578320503235, -0.5865136384963989, -1.0, 2.39541, -1.0, -1.0, -0.9631742835044861, -1.0, -0.8398264646530151, -1.0, -0.5247186422348022, -1.0, -0.8039574027061462, 4.550646, 3.1194439999999997, -0.6121099591255188, 0.04942524433135986, -1.0, -1.0, 2.39541, -0.5805400609970093, 0.729522168636322, -0.47017765045166016, -1.0, -0.8535279035568237, -0.8903297185897827, -1.0, 1.4650077819824219, -0.5201283693313599, -0.844266951084137, -0.5231894254684448, -1.0, -1.0, -1.0, 2.073133, -0.8010178804397583, 3.505625, -0.774699330329895, -1.0, 1.3387547731399536, 2.795066, -1.0, 0.8696455359458923, -0.4350496232509613, -1.0, -0.73447585105896], "live_pnl": -5.569422721862793, "live_wins": 16, "live_total": 61, "age": 127, "rounds_underwater": 50, "abstain_rounds": 0, "backtest_score": -0.5461504801460043, "backtest_acc": 0.5474814203137902, "backtest_n": 1391}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price-flow efficiency and cross-exchange consensus.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. OBI Consensus: How much the three exchanges agree on the order book imbalance
    # Since obi_spread is [0, 2], we normalize to [0, 1] where 1 is perfect agreement
    X['obi_consensus'] = 1.0 - (X['obi_spread'] / 2.0)

    # 2. Flow Efficiency: Measures how much the price moves per unit of aggressive delta.
    # High efficiency = thin market (price moves easily).
    # Low efficiency = absorption (heavy delta but price is stuck).
    # We use absolute values to capture the magnitude of the relationship.
    X['flow_efficiency'] = (X['price_vs_strike_pct'].abs() * 100.0) / (X['sum_delta'].abs() + 1e-7)

    # 3. Delta-Price Cohesion: Correlation between sign of delta and direction of price move
    X['delta_price_cohesion'] = np.sign(X['sum_delta']) * np.sign(X['price_vs_strike_pct'])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to combat 5-min noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using a Dynamic Microstructure Hurdle.
    The required Expected Value (EV) scales up when the market shows signs of 
    absorption (low efficiency) or fragmentation (low consensus).
    """
    # 1. Calculate raw Expected Value (EV) for both sides
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Extract engineered quality metrics
    consensus = features_row.get('obi_consensus', 0.5)
    efficiency = features_row.get('flow_efficiency', 1.0)
    
    # 3. Calculate Dynamic Hurdle
    # Base hurdle is 2%. 
    # We add up to 5% penalty for low consensus.
    # We add up to 5% penalty for low efficiency (absorption).
    # Low efficiency means (1 / (1 + efficiency)) approaches 1.0.
    consensus_penalty = 0.05 * (1.0 - consensus)
    efficiency_penalty = 0.05 * (1.0 / (1.0 + efficiency))
    
    dynamic_hurdle = 0.02 + consensus_penalty + efficiency_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: Reduce bet size as we approach the 5-min resolution 
        # to avoid being caught in last-second "noise" spikes.
        time_factor = max(0.4, min(1.0, seconds_remaining / 300.0))

        # Bet sizing: Scale linearly with the strength of the edge relative to the hurdle.
        # If edge is twice the hurdle, we bet more.
        bet_fraction = (best_ev / dynamic_hurdle) * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to prevent tiny, inefficient trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0