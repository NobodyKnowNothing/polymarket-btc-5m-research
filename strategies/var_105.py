# Var 105 | BT: -0.948 | Live PnL: $-6.31 | Trades: 60
# Desc: Implements "Multi-Venue Flow Congruence & Impact Efficiency (MV-FCEG)," which filters for high-conviction trades by requiring consensus across at least two exchanges and scaling bet size by the price-
# Stats: {"live_returns": [-0.4335085451602936, -1.0, -0.7622735500335693, 1.5959289073944092, 0.021347403526306152, -0.6812686920166016, -0.3134594261646271, -1.0, -0.18791727721691132, 0.0347537100315094, 0.25921201705932617, 4.550646, -0.2254883497953415, -0.35263386368751526, 0.02063661813735962, -0.689172625541687, -0.26832520961761475, -0.23606343567371368, -0.21894694864749908, -0.6434247493743896, 1.735205888748169, 0.010935768485069275, 0.3477563261985779, -0.3704933524131775, -0.5369232296943665, 1.2700715065002441, -0.7747995257377625, -0.31220996379852295, -0.8460671305656433, -0.6540875434875488, -1.0, 1.1770062446594238, -0.4762463867664337, -1.0, -1.0, -1.0, -0.8145111203193665, -0.30228114128112793, -0.5907994508743286, -1.0, -1.0, 1.8703069686889648, -0.9353918433189392, -0.39794328808784485, -0.25264203548431396, -1.0, -0.9219099879264832, 1.8077519999999998, 2.795066, -1.0, 0.04372119903564453, 1.6072020530700684, -0.26780304312705994, -0.22248311340808868, 0.1701984703540802, 0.027067720890045166, 0.014745384454727173, -1.0, 0.021575599908828735, -1.0], "live_pnl": -6.3079400062561035, "live_wins": 20, "live_total": 60, "age": 97, "rounds_underwater": 32, "abstain_rounds": 1, "backtest_score": -0.9476994132951948, "backtest_acc": 0.5453928005433553, "backtest_n": 2592}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    cross-venue consensus and the efficiency of aggressive flow.
    """
    df = df.copy()

    # 1. Multi-Venue Flow Congruence (Delta Agreement)
    # Measures the directional consensus among the three exchanges.
    # Values range from -1.0 (all bearish) to 1.0 (all bullish).
    # 0.33 or -0.33 indicates a 2-vs-1 split; 1.0 or -1.0 indicates full consensus.
    df['delta_agreement'] = (
        np.sign(df['binance_delta']) + 
        np.sign(df['coinbase_delta']) + 
        np.sign(df['kraken_delta'])
    ) / 3.0
    df['delta_agreement'] = df['delta_agreement'].fillna(0.0)

    # 2. Impact Efficiency
    # Measures how much aggressive flow (sum_delta) is driving price dispersion (price_std).
    # High efficiency suggests flow is effectively moving the market/widening spreads.
    df['impact_efficiency'] = df['sum_delta'] / (df['price_std'] + 1e-6)

    # 3. Flow Acceleration
    # The rate of change of the aggregate delta, capturing momentum shifts.
    df['flow_acceleration'] = df.groupby('market_slug')['sum_delta'].diff().fillna(0)

    # 4. Delta Dispersion
    # Measures the disagreement in the magnitude of flow across venues.
    df['delta_dispersion'] = df[['binance_delta', 'coinbase_delta', 'kraken_delta']].std(axis=1).fillna(0.0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to avoid overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade using "Multi-Venue Flow Congruence & Impact Efficiency (MV-FCEG)".

    Logic:
    - Consensus Gate: We only trade if the majority of exchanges (at least 2/3) 
      agree on the direction of the aggressive flow.
    - Asymmetric Thresholds: If we trade with the consensus, we use a lower EV threshold.
      If we trade against the consensus (counter-trend), we require much higher EV.
    - Dynamic Scaling: Bet size is scaled by the strength of the consensus, 
      the efficiency of the flow, and time decay.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    agreement = features_row.get('delta_agreement', 0.0)
    efficiency = features_row.get('impact_efficiency', 0.0)

    # 1. Consensus Gate
    # If agreement is 0 (e.g., one pos, one neg, one zero), it's too noisy.
    if abs(agreement) < 0.33:
        return None, 0.0

    decision_side = None
    best_ev = 0.0

    if agreement > 0:  # Majority Bullish
        if ev_up > 0.03:
            decision_side = 'Up'
            best_ev = ev_up
        elif ev_down > 0.05:  # High bar for counter-trend
            decision_side = 'Down'
            best_ev = ev_down
    else:  # Majority Bearish
        if ev_down > 0.03:
            decision_side = 'Down'
            best_ev = ev_down
        elif ev_up > 0.05:  # High bar for counter-trend
            decision_side = 'Up'
            best_ev = ev_up

    if decision_side is None:
        return None, 0.0

    # 2. Bet Sizing Logic
    # Base scale on EV relative to a 5% edge
    base_bet = best_ev / 0.05
    
    # Multiplier 1: Consensus Strength (1.0 for full consensus, ~0.66 for 2/3 split)
    consensus_mult = abs(agreement)
    
    # Multiplier 2: Impact Efficiency (higher efficiency = cleaner move)
    # We cap the efficiency multiplier to prevent extreme bets on low-liquidity spikes.
    eff_mult = 1.0 + min(1.0, abs(efficiency) / 10.0)
    
    # Multiplier 3: Time Decay (reduce exposure as the window closes)
    time_decay = min(1.0, seconds_remaining / 300.0)

    bet_fraction = base_bet * consensus_mult * eff_mult * time_decay

    return decision_side, min(1.0, max(0.0, bet_fraction))