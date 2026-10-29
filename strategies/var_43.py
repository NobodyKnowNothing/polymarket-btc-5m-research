# Var 43 | BT: 0.046 | Live PnL: $-3.59 | Trades: 61
# Desc: Implements "Multi-Exchange Flow Convergence (MEFC)," which filters for high-conviction moves by requiring simultaneous agreement in aggressive delta across all three exchanges and alignment between th
# Stats: {"live_returns": [-0.29631340503692627, -0.09418610483407974, -0.16532564163208008, -0.06869153678417206, 0.1854453682899475, -0.3518902659416199, -0.22364340722560883, -0.11130113899707794, 0.33426836133003235, -0.3261226415634155, -0.2854853570461273, -0.20282413065433502, -0.12677937746047974, -0.11064804345369339, 0.006166212260723114, -0.3242867887020111, 0.004869300872087479, -0.17847125232219696, -0.11436890065670013, 0.07012977451086044, -0.0814221054315567, 0.19626930356025696, 0.8889255523681641, -0.1871766299009323, 0.31064313650131226, -0.09812668710947037, -0.08413253724575043, -0.16279496252536774, 1.4481120109558105, -0.19401080906391144, 0.3471642732620239, -0.0852573812007904, -0.166519433259964, -0.06372125446796417, -0.11928088963031769, -0.12549366056919098, -0.09179176390171051, 0.3136484622955322, -0.15813782811164856, -0.23887494206428528, -0.11412826925516129, -0.24639655649662018, -0.09413778781890869, -0.13956092298030853, -0.06846679747104645, -0.3121512532234192, -0.41100701689720154, 0.3442767262458801, 0.006863921880722046, -0.18421295285224915, -0.5588679909706116, -0.347989022731781, -0.08547549694776535, -0.2317236214876175, -0.6023141145706177, 0.39888229966163635, -0.14913524687290192, 0.4323907494544983, -0.09726431220769882, -0.318634569644928, -0.0813383162021637], "live_pnl": -3.591827630996704, "live_wins": 15, "live_total": 61, "age": 104, "rounds_underwater": 30, "abstain_rounds": 0, "backtest_score": 0.0458051869310242, "backtest_acc": 0.5685468808367575, "backtest_n": 2012}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-exchange delta consensus and flow-book alignment.
    """
    df = df.copy()

    # 1. Delta Consensus: How much do the three exchanges agree on the direction of aggressive flow?
    # High absolute value = all exchanges moving in the same direction.
    # Low absolute value = exchanges are fighting or delta is localized to one exchange.
    delta_cols = ['binance_delta', 'coinbase_delta', 'kraken_delta']
    delta_std = df[delta_cols].std(axis=1)
    df['delta_consensus'] = df['mean_delta'] / (delta_std + 1e-6)

    # 2. OBI Consensus: Do the order books across exchanges show a unified pressure?
    # Uses the provided obi_spread to gauge disagreement.
    df['obi_consensus'] = df['mean_obi'] / (df['obi_spread'] + 1e-6)

    # 3. Flow-Book Alignment: Is the aggressive flow (Delta) supported by the book (OBI)?
    # Positive = Flow and OBI are in sync (e.g., buying and more bids).
    # Negative = Flow and OBI are fighting (e.g., buying into heavy asks/absorption).
    df['flow_book_alignment'] = df['mean_delta'] * df['mean_obi']

    # 4. Delta Acceleration: Is the aggressive flow speeding up or slowing down?
    # Group by market_slug to prevent leakage.
    df['delta_accel'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 5. Volatility-Adjusted Aggression: Is the delta significant relative to current volatility?
    df['vol_delta_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target columns
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters."""
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
    Decide whether to trade using "Multi-Exchange Flow Convergence (MEFC)".
    The strategy seeks moments where aggressive flow is unanimous across exchanges
    and is actively supported by the order book imbalance.
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
    delta_consensus = features_row.get('delta_consensus', 0.0)
    obi_consensus = features_row.get('obi_consensus', 0.0)
    flow_book_alignment = features_row.get('flow_book_alignment', 0.0)
    delta_accel = features_row.get('delta_accel', 0.0)

    # 3. Dynamic Hurdle
    # We require a higher EV if the market is in a state of high disagreement.
    base_hurdle = 0.025
    # If delta_consensus is low, the hurdle increases.
    consensus_penalty = max(0, 0.02 - abs(delta_consensus) * 0.01)
    hurdle = base_hurdle + consensus_penalty

    if best_ev < hurdle:
        return None, 0.0

    # 4. Multiplier Logic (The "Convergence" Filter)
    multiplier = 1.0

    # A) Flow Consensus: Is the aggression unanimous?
    # If exchanges are fighting (low consensus), heavily reduce bet size.
    if abs(delta_consensus) < 0.8:
        multiplier *= 0.3
    elif abs(delta_consensus) > 2.0:
        multiplier *= 1.2  # Bonus for extreme consensus

    # B) Book Alignment: Is the book supporting the move?
    # If we are betting 'Up' but alignment is negative, we are hitting a wall.
    if (side == 'Up' and flow_book_alignment < 0) or (side == 'Down' and flow_book_alignment > 0):
        multiplier *= 0.4

    # C) OBI Consensus: Is the book pressure unified?
    if abs(obi_consensus) < 0.5:
        multiplier *= 0.6

    # D) Momentum Confirmation: Is the delta accelerating in our direction?
    if (side == 'Up' and delta_accel < 0) or (side == 'Down' and delta_accel > 0):
        multiplier *= 0.7

    # 5. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid last-second volatility spikes.
    time_factor = (seconds_remaining / 300.0) ** 1.5

    # Final bet fraction: Scale by (Edge/Hurdle) and the convergence multiplier
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure trades are worth the potential slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction