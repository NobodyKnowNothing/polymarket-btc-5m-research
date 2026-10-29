# Var 28 | BT: -0.809 | Live PnL: $-8.42 | Trades: 61
# Desc: Implements "High-Cohesion Microstructure Capture," which uses price consensus (inverse of cross-exchange dispersion) and delta-OBI alignment to filter out fragmented or absorbing price moves, while sc
# Stats: {"live_returns": [-0.16937224566936493, 0.6675275564193726, -0.4974357783794403, -0.18008080124855042, -0.1512082815170288, -0.2670540511608124, -0.22628134489059448, 1.4071025848388672, -0.14046092331409454, -0.2954035699367523, -0.7240030765533447, -0.1055656224489212, -0.45086726546287537, -0.2377764731645584, -0.40451809763908386, -0.1354704052209854, -0.1460210233926773, -0.3204169273376465, -0.21858729422092438, 0.5869215726852417, 0.30266883969306946, -0.11036553233861923, -0.2120000720024109, 0.024075359106063843, -1.0, -0.3523472249507904, -0.18812185525894165, -0.3995233178138733, -0.4371269941329956, -0.33851146697998047, 0.17109644412994385, -0.40444982051849365, 4.113521575927734, -1.0, 2.8658273220062256, -0.16673605144023895, -0.5568857192993164, -0.9292558431625366, -0.278138667345047, -1.0, -0.43601131439208984, -0.9090732932090759, -0.7111459970474243, 0.2576850950717926, -0.6541172862052917, -1.0, -0.3568548560142517, -0.3315913677215576, -0.4036506712436676, -0.6427050828933716, 0.661167562007904, -1.0, -1.0, -1.0, -1.0, 0.01199348270893097, 1.4679861068725586, 0.0308743417263031, -0.6798028945922852, 1.5872235298156738, -0.4063011407852173], "live_pnl": -8.41956901550293, "live_wins": 14, "live_total": 61, "age": 118, "rounds_underwater": 24, "abstain_rounds": 0, "backtest_score": -0.808998635072961, "backtest_acc": 0.5522203947368421, "backtest_n": 1532}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price consensus, flow-book alignment, and price-to-strike drift.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Price Consensus: How much do the three exchanges agree?
    # High price_std means fragmentation; low price_std means consensus.
    X['price_consensus'] = 1.0 / (1.0 + X['price_std'])

    # 2. Flow-Book Alignment: Do aggressive trades (delta) match the book imbalance (OBI)?
    # Positive value means delta and OBI are in the same direction (trending).
    # Negative value means delta and OBI are opposing (absorption/exhaustion).
    X['flow_alignment'] = X['mean_delta'] * X['mean_obi']

    # 3. Price Drift: Where is the current price relative to the market start?
    X['price_drift'] = (X['binance_price'] - X['strike_price']) / (X['strike_price'] + 1e-7)

    # 4. Volatility-Adjusted Momentum: Momentum scaled by the current volatility regime
    X['vol_adj_momentum'] = X['mean_momentum'] / (X['mean_volatility'] + 1e-7)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 25.0,
        'reg_lambda': 50.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "High-Cohesion" logic.
    We only enter when there is high agreement across exchanges (consensus) 
    and the aggressive flow is not being absorbed by the order book.
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
    consensus = features_row.get('price_consensus', 0.5)
    alignment = features_row.get('flow_alignment', 0.0)
    
    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 2.5%
    hurdle = 0.025

    # Penalty A: Absorption/Exhaustion
    # If alignment is negative, delta and OBI are fighting. This suggests absorption.
    # We increase the hurdle significantly to avoid "fake" moves.
    if alignment < 0:
        hurdle += 0.04

    # Penalty B: Fragmentation
    # If consensus is low (high price_std), the market is fragmented.
    # We increase the hurdle to demand a higher edge during uncertainty.
    if consensus < 0.8:
        hurdle += 0.03

    # 4. Execution Logic
    if best_ev > hurdle:
        # Time-decay: Use a square-root decay to reduce exposure as the window closes.
        # This protects against the high-volatility "noise" in the final seconds.
        time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

        # Confidence Scaling: Scale bet size by consensus (agreement) and edge magnitude.
        # A high consensus (near 1.0) allows for a larger bet.
        bet_fraction = (best_ev / hurdle) * consensus * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure trades are worth the risk
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0