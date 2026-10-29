# Var 66 | BT: 0.074 | Live PnL: $-6.74 | Trades: 61
# Desc: Implements "Microstructure-Weighted Expected Value (MWEV)," which replaces rigid logic gates with a continuous scaling mechanism that weights the model's edge by the efficiency of the price-flow relat
# Stats: {"live_returns": [0.05164444446563721, -1.0, 1.8077519999999998, -1.0, 0.16389999999999993, 0.5293476581573486, -0.25858810544013977, 0.02501872181892395, 2.651597, 2.280573, -1.0, -1.0, -0.7274806499481201, -1.0, -1.0, 1.7754337787628174, -1.0, -1.0, -1.0, -1.0, 0.019518733024597168, -1.0, -1.0, -1.0, 5.282196, -1.0, -1.0, -0.05160016939043999, -1.0, 4.550646, 3.1194439999999997, -1.0, -1.0, -1.0, -1.0, -0.4578936696052551, -1.0, 2.39541, 0.020821988582611084, -0.36308932304382324, 0.5701698064804077, 0.9852772951126099, -1.0, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, -0.9178050756454468, -0.9283739328384399, -1.0, 0.028777673840522766, -1.0, -0.8512730002403259, -1.0, 0.017460763454437256, -1.0, -0.5132326483726501, -1.0, 2.073133, -0.998438835144043], "live_pnl": -6.740512371063232, "live_wins": 20, "live_total": 61, "age": 102, "rounds_underwater": 13, "abstain_rounds": 0, "backtest_score": 0.07409925525679778, "backtest_acc": 0.571390158644797, "backtest_n": 3058}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the efficiency of price movement relative to aggressive flow 
    and the level of consensus across exchanges.
    """
    df = df.copy()

    # 1. Price Change: Recent velocity of the primary exchange
    df['price_change'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Delta-Price Efficiency: How much price moves per unit of aggressive delta.
    # High values mean efficient trends; low/negative values suggest absorption.
    # We clip this to prevent extreme outliers from destabilizing the XGBoost model.
    df['delta_price_efficiency'] = (df['price_change'] / (df['mean_delta'].abs() + 1e-6)).clip(-10, 10)

    # 3. OBI Agreement: A measure of how much the three exchanges agree on liquidity.
    # Since obi_spread is max_obi - min_obi (range 0 to 2), we normalize to 0-1.
    df['obi_agreement'] = 1.0 - (df['obi_spread'] / 2.0)

    # 4. Momentum Alignment: Does the aggressive flow (Delta) match the price direction?
    # 1.0 if they align, 0.5 if they are neutral or opposing (absorption risk).
    df['delta_sign_match'] = np.where(df['mean_delta'] * df['price_change'] > 0, 1.0, 0.5)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Weighted Expected Value (MWEV)".

    Logic:
    Instead of hard "if/else" gates, we calculate a continuous bet_fraction.
    The bet size is the Model's Edge (EV) scaled by:
    - The consensus of OBI across exchanges (obi_agreement)
    - The efficiency of delta driving price (delta_sign_match)
    
    This allows for large bets on massive mispricings even in slightly messy 
    microstructures, but heavily penalizes trades where the market is 
    absorbing flow or exchanging disagreement is high.
    """
    # 1. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # 2. Time-gate: Avoid the extreme volatility/noise of the final 30 seconds
    if seconds_remaining < 30:
        return None, 0.0

    # 3. Minimum Edge Threshold
    # We only enter if the model finds a meaningful mispricing.
    if best_ev < 0.025:
        return None, 0.0

    # 4. Microstructure Multiplier
    # We extract our engineered continuous signals
    agreement = features_row.get('obi_agreement', 1.0)
    alignment = features_row.get('delta_sign_match', 1.0)
    
    # The multiplier scales the bet based on how "clean" the microstructure is.
    # If agreement is 0.5 and alignment is 0.5, the bet is 25% of what it would be.
    multiplier = agreement * alignment

    # 5. Dynamic Bet Sizing
    # Scale bet_fraction linearly with EV, then apply the microstructure multiplier.
    # We use 0.04 as a normalization constant (expected "strong" edge).
    bet_fraction = (best_ev / 0.04) * multiplier

    # Clamp bet_fraction to [0.0, 1.0]
    bet_fraction = max(0.0, min(1.0, bet_fraction))

    if bet_fraction > 0:
        return side, bet_fraction

    return None, 0.0