# Var 177 | BT: 2.012 | Live PnL: $0.00 | Trades: 0
# Desc: variant 1
# Stats: {"live_returns": [], "live_pnl": 0.0, "live_wins": 0, "live_total": 0, "age": 60, "rounds_underwater": 0, "abstain_rounds": 60, "backtest_score": 2.011614415092236, "backtest_acc": 0.5513384671800513, "backtest_n": 51}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the directional congruence (coherence) of price, intent, and flow.
    """
    df = df.copy()

    # 1. Price Momentum Direction (using Binance as lead)
    # We group by market_slug to ensure diffs don't leak across different markets.
    df['price_diff'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['price_dir'] = np.sign(df['price_diff'])

    # 2. Intent Direction (OBI)
    df['obi_dir'] = np.sign(df['mean_obi'])

    # 3. Flow Direction (Delta)
    df['delta_dir'] = np.sign(df['mean_delta'])

    # 4. Coherence Score: The average alignment of Price, OBI, and Delta.
    # Range [-1, 1]. 1.0 = perfect alignment, 0 = neutral, -1.0 = perfect opposition.
    df['coherence'] = (df['price_dir'] + df['obi_dir'] + df['delta_dir']) / 3.0

    # 5. Dispersion Index: Cross-exchange price disagreement normalized by volatility.
    # High dispersion indicates market instability or arbitrage opportunities.
    df['dispersion'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 6. Funding-Flow Alignment: Do aggressive traders align with the funding rate?
    df['funding_flow_alignment'] = np.sign(df['mean_funding']) * np.sign(df['mean_delta'])

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: Handle potential infs from division and NaNs from diffs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return hyperparameters optimized for capturing regime-based congruence."""
    return {
        'n_estimators': 1000,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 5.0,
        'reg_lambda': 5.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Vector Congruence & Dispersion Gating (TVCDG)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Apply a Dynamic Hurdle: As time runs out, the required EV increases to 
       avoid late-window noise.
    3. Dispersion Gate: If the price spread across exchanges is too high 
       relative to volatility, the market is too unstable to trade.
    4. Congruence Scaling:
       - If the model's predicted direction aligns with the Coherence Score 
         (Price, OBI, and Delta all pointing the same way), scale bet size up.
       - If the model's direction contradicts the microstructure (Divergence), 
         only take the trade if the EV is extremely high (potential reversal).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # Extract engineered features
    coherence = features_row.get('coherence', 0.0)
    dispersion = features_row.get('dispersion', 0.0)

    # 1. Dynamic EV Hurdle
    # Starts at 3% at the beginning of the 5-min window, rises to 5% at the end.
    time_decay = seconds_remaining / 300.0
    hurdle = 0.03 + (1.0 - time_decay) * 0.02

    # 2. Dispersion Gate
    # If exchanges disagree significantly (dispersion > 2.0), abstain.
    if dispersion > 2.0:
        return None, 0.0

    # 3. Execution Logic
    if best_ev > hurdle:
        # Check if the model's direction is congruent with the microstructure vectors
        is_congruent = (side == 'Up' and coherence > 0) or (side == 'Down' and coherence < 0)
        
        if is_congruent:
            # Scale bet based on the strength of the congruence
            # A perfect coherence (1.0) gives full sizing relative to EV.
            target_ev = 0.04
            bet_fraction = (best_ev / target_ev) * (0.5 + 0.5 * coherence)
            return side, min(1.0, max(0.1, bet_fraction))
        
        elif best_ev > 0.06:
            # Divergence Trade: The model says one thing, but flow/OBI say another.
            # Only take this if the edge is massive (potential exhaustion/reversal).
            return side, 0.15

    return None, 0.0