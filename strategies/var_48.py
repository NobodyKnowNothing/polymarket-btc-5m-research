# Var 48 | BT: -0.669 | Live PnL: $-7.83 | Trades: 61
# Desc: Implements "Funding-Delta Divergence (FDD)," which identifies high-conviction squeeze setups by scaling bet sizes when aggressive trade flow (Delta) and perp funding rates move in opposite directions,
# Stats: {"live_returns": [-1.0, -0.8181334137916565, 1.28551, -1.0, 4.245841, -1.0, -0.16795532405376434, -0.5953046679496765, -0.833938479423523, -1.0, 2.280573, -0.6835276484489441, -1.0, -0.31111836433410645, 1.6783945560455322, 0.6310768127441406, -0.8534830212593079, 1.5191249999999998, -1.0, -1.0, 1.5854279999999998, 1.3964720000000002, -1.0, -0.6341586112976074, -0.5368824601173401, -1.0, -0.6588966846466064, -1.0, -0.497213214635849, -1.0, -0.16796188056468964, 0.05151817202568054, 0.2569270133972168, -1.0, 4.550646, -0.5604431629180908, 3.1194439999999997, -1.0, -1.0, -0.6188333630561829, -1.0, -0.5594484806060791, -0.6212413311004639, -1.0, -0.6458652019500732, 2.39541, 0.42319920659065247, -1.0, -1.0, 0.1591700166463852, 0.847965657711029, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -0.471413791179657, -1.0], "live_pnl": -7.829977035522461, "live_wins": 17, "live_total": 61, "age": 81, "rounds_underwater": 21, "abstain_rounds": 0, "backtest_score": -0.6691562519944425, "backtest_acc": 0.5543438749545619, "backtest_n": 2320}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between aggressive flow, order book pressure, and 
    funding-driven squeeze potential.
    """
    df = df.copy()

    # 1. Delta-OBI Congruence: Do aggressive trades align with book pressure?
    # Positive = Flow and Book are in sync; Negative = Flow is fighting the book (Absorption).
    df['delta_obi_congruence'] = df['mean_delta'] * df['mean_obi']

    # 2. Funding-Delta Divergence: Measures potential for a squeeze.
    # Highly negative values indicate a "Long Squeeze" (Positive funding, Negative delta).
    # Highly positive values indicate a "Short Squeeze" (Negative funding, Positive delta).
    df['funding_delta_divergence'] = df['mean_funding'] * df['mean_delta']

    # 3. Delta Efficiency: How much price dispersion is generated per unit of delta.
    # Low values suggest "Absorption" (High delta, low price movement).
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

    # 4. Cross-Exchange Price Tension: How much are the exchanges disagreeing?
    df['exchange_tension'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Funding-Delta Divergence (FDD)".
    The strategy seeks to capitalize on 'squeezes' by identifying when 
    aggressive flow (Delta) is moving against the funding-implied bias, 
    and scales exposure heavily when the model's direction aligns with that squeeze.
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
    funding = features_row.get('mean_funding', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    congruence = features_row.get('delta_obi_congruence', 0.0)
    efficiency = features_row.get('delta_efficiency', 0.0)

    # 3. Base Hurdle
    # We require a minimum edge to overcome the noise of a 5-minute window.
    hurdle = 0.025
    if best_ev < hurdle:
        return None, 0.0

    # 4. Squeeze-Regime Multiplier
    # We look for divergence between funding and delta.
    multiplier = 1.0
    
    # Case A: Long Squeeze Potential (Funding is positive, but Delta is negative)
    if funding > 0.0001 and delta < -0.01:
        if side == 'Down':
            multiplier = 1.8  # High conviction: Model predicts the squeeze direction
        else:
            multiplier = 0.2  # Avoid: Model is trying to catch a falling knife

    # Case B: Short Squeeze Potential (Funding is negative, but Delta is positive)
    elif funding < -0.0001 and delta > 0.01:
        if side == 'Up':
            multiplier = 1.8  # High conviction: Model predicts the squeeze direction
        else:
            multiplier = 0.2  # Avoid: Model is trying to catch a mooning rocket

    # 5. Absorption Penalty
    # If delta and OBI are fighting (congruence is negative), the move is likely being absorbed.
    if (side == 'Up' and congruence < 0) or (side == 'Down' and congruence > 0):
        multiplier *= 0.5
    
    # If delta is high but price isn't moving (low efficiency), reduce confidence.
    if abs(efficiency) < 0.1:
        multiplier *= 0.6

    # 6. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid late-window volatility.
    time_factor = (seconds_remaining / 300.0)
    
    # Scale bet_fraction by (Edge / Hurdle) and the regime multiplier.
    # We cap the bet_fraction at 1.0.
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Final safety check: if the multiplier or edge is too low, abstain.
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction