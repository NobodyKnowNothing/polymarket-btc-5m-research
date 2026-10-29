# Var 207 | BT: 0.418 | Live PnL: $-1.52 | Trades: 60
# Desc: Implements "Regime-Adaptive Signal Weighting (RASW)," which dynamically shifts the required signal type (Order Book Intent vs. Aggressive Flow) and entry hurdles based on the current Volatility and Fu
# Stats: {"live_returns": [0.1979141, -0.16345084603627522, 0.33705840472127285, -0.20738732115817068, 0.056238897340965786, -0.17654759496355052, 0.22559383672894617, -0.3289728376070658, -0.15125269360091156, -0.1633323136170705, -0.16389197679360706, -0.1, -0.11895963373435868, 0.18908480000000003, -0.1631651744445165, -0.055732163815561925, 0.8591705987604925, -0.1, -0.1609795108238856, -0.1621815980275472, -0.19041307540800306, 0.02148396470408087, 0.13514172203298253, -0.2334557714537091, -0.1, -0.1, -0.1, -0.1, -0.3008071618080139, 0.3769319332633274, -0.13721047027309208, -0.17223960557195872, -0.1, -0.1, -0.21116418457321592, -0.20773332008050277, -0.10167711287657419, 0.24483974037027587, -0.23093988477563857, -0.3025027310912609, -0.1, 0.1807752, 0.23198240163645148, -0.26870751690864564, -0.2952896300135718, -0.25041230088357924, -0.3126853291988373, -0.30710287555058796, 0.424357170834396, -0.3154132782618205, 0.19006716462687756, -0.1, 0.34398278513654873, 0.5621364195281641, -0.18106380099786654, -0.1, 1.3571411224698358, -0.1615139539639155, -0.32423650813102717, 0.167220994795475], "live_pnl": -1.51930091949425, "live_wins": 18, "live_total": 60, "age": 100, "rounds_underwater": 43, "abstain_rounds": 0, "backtest_score": 0.4181284673943725, "backtest_acc": 0.5630779968591869, "backtest_n": 4113}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    regime-specific signal interactions (Intent vs. Action).
    """
    df = df.copy()

    # 1. Signal Alignment & Velocity
    # Binary alignment: 1.0 if OBI and Delta point in the same direction, else 0.0
    df['obi_delta_alignment'] = ((np.sign(df['mean_obi']) == np.sign(df['mean_delta'])) & 
                                 (df['mean_obi'] != 0)).astype(float)
    
    # Velocity of the signals (rate of change within the market window)
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    
    # 2. Regime-Interaction Features
    # How much funding is present relative to current volatility
    df['funding_vol_interaction'] = df['mean_funding'] * df['mean_volatility']
    
    # Efficiency of flow: How much price moves per unit of delta
    # High efficiency = Liquidity Vacuum; Low efficiency = Absorption
    df['flow_efficiency'] = df['mean_delta'].apply(lambda x: 0 if abs(x) < 1e-6 else 0) # placeholder logic
    # Correct way to do it in vectorized form:
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['flow_efficiency'] = df['price_change'] / (df['mean_delta'] + 1e-6)

    # 3. Cross-Exchange Divergence
    # Measures how much the exchanges disagree on the OBI
    df['obi_consensus_score'] = 1.0 / (1.0 + df['obi_spread'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on strong regime signals."""
    return {
        'n_estimators': 800,
        'max_depth': 3,
        'learning_rate': 0.015,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Regime-Adaptive Signal Weighting (RASW)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Regime Detection:
       - 'Coiling' (Low Vol): Market is quiet; prioritize Order Book Intent (OBI).
       - 'Trending' (High Vol): Market is moving; prioritize Aggressive Flow (Delta).
       - 'Extreme' (High Funding): Leverage is high; reduce hurdles to capture squeezes.
    3. Adaptive Hurdle:
       - In Coiling regimes, require high OBI alignment and consensus.
       - In Trending regimes, require high Delta velocity and price efficiency.
    4. Dynamic Sizing:
       - Scale bet size by the 'obi_delta_alignment' and the magnitude of the signal.
    """
    # Exit if too little time remains to capture a move
    if seconds_remaining < 45:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
        signal_val = features_row.get('mean_obi', 0) if features_row.get('mean_volatility', 0) < 0.001 else features_row.get('mean_delta', 0)
    else:
        side = 'Down'
        best_ev = ev_down
        signal_val = features_row.get('mean_obi', 0) if features_row.get('mean_volatility', 0) < 0.001 else features_row.get('mean_delta', 0)

    # Extract features for decision making
    volatility = features_row.get('mean_volatility', 0.0)
    funding = features_row.get('mean_funding', 0.0)
    obi = features_row.get('mean_obi', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    alignment = features_row.get('obi_delta_alignment', 0.5)
    consensus = features_row.get('obi_consensus_score', 0.5)

    # --- 1. DYNAMIC HURDLE CALCULATION ---
    hurdle = 0.03

    # Regime: Coiling (Low Volatility)
    if volatility < 0.0012:
        # In quiet markets, we only trust signals where OBI and Delta agree
        if alignment < 1.0:
            hurdle += 0.03
        # And where the intent (OBI) is actually significant
        if abs(obi) < 0.15:
            hurdle += 0.02
            
    # Regime: Trending (High Volatility)
    elif volatility > 0.0025:
        # In fast markets, we only trust signals where Delta is aggressive
        if abs(delta) < 0.3:
            hurdle += 0.03
        # And where consensus across exchanges is high
        if consensus < 0.7:
            hurdle += 0.02

    # Regime: Leverage/Squeeze (High Funding)
    if abs(funding) > 0.0005:
        # Funding provides extra conviction for directional moves
        hurdle -= 0.015

    # --- 2. EXECUTION ---
    if best_ev > hurdle:
        # Base bet size is scaled by alignment and the strength of the signal
        # We want to bet more when the model and the microstructure agree.
        signal_strength = min(1.0, abs(signal_val) * 2)
        bet_fraction = (alignment * 0.5 + 0.5) * 0.7 * signal_strength
        
        # Time decay: Reduce size as the window closes to avoid late-entry slippage
        time_factor = min(1.0, seconds_remaining / 300.0)
        final_bet = min(1.0, bet_fraction * time_factor)

        # Safety floor for high-conviction trades
        if final_bet < 0.1 and best_ev > (hurdle + 0.05):
            final_bet = 0.1

        return side, final_bet

    return None, 0.0