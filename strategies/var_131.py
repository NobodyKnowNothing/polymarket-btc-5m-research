# Var 131 | BT: 0.638 | Live PnL: $-6.40 | Trades: 61
# Desc: Implements "Microstructure Congruence & Volatility-Adaptive Scaling (MCVAS)," which distinguishes between trend-reinforcing flow (congruent Delta and OBI) and exhaustion/absorption regimes (divergent
# Stats: {"live_returns": [2.280573, -1.0, -1.0, -1.0, -1.0, -0.3790872395038605, -0.6835780143737793, -1.0, -1.0, -1.0, 2.795066, -0.8681672215461731, -0.5915789604187012, -1.0, -1.0, -1.0, -1.0, 1.2637920379638672, -1.0, -1.0, -0.6125527024269104, 0.060593366622924805, 1.7294070000000001, -1.0, -1.0, -1.0, -0.6832433342933655, -1.0, 1.1074661016464233, -1.0, -1.0, -0.8028481602668762, 2.39541, 0.98679518699646, 2.280573, 1.946404218673706, -1.0, 1.890848, -0.8907168507575989, -1.0, -1.0, 1.3396370000000002, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -1.0, 3.727685, -1.0, -0.6552877426147461, 2.39541, -1.0, -1.0, -1.0, -1.0, -0.9180493950843811, -1.0, 0.10316799999999993, -0.9759547114372253, -1.0, -1.0], "live_pnl": -6.396245002746582, "live_wins": 19, "live_total": 61, "age": 98, "rounds_underwater": 8, "abstain_rounds": 0, "backtest_score": 0.6377755605609547, "backtest_acc": 0.5537840565085772, "backtest_n": 3354}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the congruence between aggressive flow (Delta) and passive depth (OBI).
    """
    df = df.copy()

    # 1. Delta-OBI Congruence: Do aggressive trades align with order book imbalance?
    # 1.0 = Trend (Both buying/selling), -1.0 = Absorption (Delta vs OBI mismatch), 0.0 = Neutral
    df['delta_obi_congruence'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 2. Delta-OBI Intensity: The combined magnitude of the flow and depth signals.
    df['delta_obi_intensity'] = abs(df['mean_delta'] * df['mean_obi'])

    # 3. Volatility-Dispersion: Captures the 'noise' level by combining volatility and price spread.
    df['vol_price_dispersion'] = df['mean_volatility'] * df['price_std']

    # 4. Price Velocity (Relative to Strike): How fast is the price moving away from the resolution point?
    df['velocity_to_strike'] = (df['binance_price'] - df['strike_price']) / (df['seconds_remaining'] + 1e-6)

    # Drop metadata and target columns
    # Note: 'seconds_remaining' is metadata and should not be in X, 
    # but it is passed separately to the strategy_filter.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust microstructure signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Congruence & Volatility-Adaptive Scaling (MCVAS)".

    Logic:
    - Calculate raw EV for both sides.
    - Congruence Multiplier: 
        - If Delta and OBI agree (congruence=1), amplify EV (Trend reinforcement).
        - If Delta and OBI disagree (congruence=-1), heavily penalize EV (Absorption/Exhaustion).
        - If neutral, apply a slight discount.
    - Volatility/Time Scaling:
        - Reduce bet size as volatility increases to manage variance.
        - Reduce bet size as time runs out to avoid last-second noise.
    - Threshold: Only trade if the 'effective' EV (after microstructure adjustment) is significant.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        base_ev = ev_up
    else:
        side = 'Down'
        base_ev = ev_down

    # Extract engineered features
    congruence = features_row.get('delta_obi_congruence', 0.0)
    intensity = features_row.get('delta_obi_intensity', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    # 1. Congruence Multiplier
    # If trend is reinforced (Delta and OBI agree), we increase confidence.
    # If trend is being absorbed (Delta and OBI disagree), we treat the signal as high-risk.
    if congruence > 0:
        multiplier = 1.2 + (0.4 * intensity)  # Trend reinforcement
    elif congruence < 0:
        multiplier = 0.35                     # Absorption/Exhaustion penalty
    else:
        multiplier = 0.75                     # Neutral/Uncertainty discount

    effective_ev = base_ev * multiplier

    # 2. Risk Scaling Factors
    # Volatility Penalty: Higher volatility = lower bet fraction
    vol_adj = 1.0 / (1.0 + mean_vol)
    
    # Time Decay: Avoid high-variance bets in the final 60 seconds
    time_adj = min(1.0, seconds_remaining / 180.0)

    # 3. Execution Decision
    # We use a higher effective EV threshold to ensure the microstructure supports the model.
    entry_threshold = 0.03
    
    if effective_ev > entry_threshold:
        # Scale bet fraction based on effective edge, volatility, and time remaining.
        # Target a 4% effective edge for a full bet.
        bet_fraction = (effective_ev / 0.04) * vol_adj * time_adj
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0