# Var 197 | BT: 0.460 | Live PnL: $-0.10 | Trades: 66
# Desc: Implements "Microstructure Signal Stability (MSS)," which scales trade conviction and entry hurdles based on the temporal consistency of the alignment between Order Book Imbalance (OBI) and Aggressive
# Stats: {"live_returns": [0.997918403845588, 0.11977050000000002, -0.5289427986070084, -0.26452079963320785, 0.8319610418717569, -0.39848087844244456, 1.4214049463979443, 0.5571278329596809, 0.030917578959721914, 1.9538655377600074, 0.5339311509316277, 0.9178651089133179, -0.13197987247361231, -0.5314385558004208, -0.6634100033123823, -0.519022016132422, -0.39480172666667845, -0.39905398329324837, -0.666131223217402, -0.39722319441157633, 1.155403352789944, -0.3990996182672409, 1.921968832884747, -0.26662874485764404, 0.9039930199560222, -0.3974241623026163, -0.39876781863719013, -0.3992539529118056, 0.05973194742279753, 1.6489616695420894, -0.25902008481819666, -0.5234250034462938, 0.524943327837828, -0.1330028989944027, 0.7531164683833064, -0.3945286840013478, -0.3974350666919099, 0.6015690602601247, -0.3949048802347784, -0.6563106497282168, -0.6555312515376418, -0.3985475839080007, -0.26371705064942275, 0.24630950127222276, -0.5291308694694392, 1.0546889021495216, -0.26590368457219526, -0.5288104943239003, 0.0541988095017239, 0.012218557161608856, 0.08538111750869937, -0.5324625451888346, 0.025128790923481636, -0.7929778299135364, -0.526407913649703, 0.042248601751385784, -0.13226092076838553, -0.263765850765069, -0.26365315929252, 0.3168565668393538, -0.399703053531534, 0.05990969825032977, -0.39953142232372696, -0.530508701918739, -0.39772419429366107, -0.5323028204570944], "live_pnl": -0.0963556373706182, "live_wins": 26, "live_total": 66, "age": 124, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": 0.4599314468340601, "backtest_acc": 0.5378613229295974, "backtest_n": 3701}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on the 
    temporal stability of the consensus between order book intent and aggressive flow.
    """
    df = df.copy()

    # 1. Directional Alignment
    # We calculate the sign of both OBI and Delta to see if they agree.
    df['delta_sign'] = np.sign(df['mean_delta'])
    df['obi_sign'] = np.sign(df['mean_obi'])
    
    # Alignment is 1.0 if both signs match and are non-zero, else 0.0
    df['alignment'] = ((df['delta_sign'] == df['obi_sign']) & (df['delta_sign'] != 0)).astype(float)

    # 2. Signal Stability (Temporal Consistency)
    # We use a rolling mean of the alignment to see how "stable" the consensus is.
    # A high alignment_rolling means the market has been in a consistent state for several snapshots.
    df['alignment_rolling'] = df.groupby('market_slug')['alignment'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )

    # 3. Volatility-Normalized Flow
    # How much aggressive flow is occurring relative to the current volatility regime.
    df['delta_vol_ratio'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return regularized hyperparameters to prevent overfitting to noise in the stability signals."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Signal Stability (MSS)".

    Logic:
    1. Identify Edge: Calculate EV for both sides.
    2. Dynamic Hurdle: 
       - Base hurdle is 0.03.
       - If the signal is "flickering" (low alignment_rolling), increase the hurdle significantly.
       - If volatility is high, increase the hurdle to account for noise.
    3. Dynamic Sizing:
       - Scale bet size by the stability (alignment_rolling).
       - Apply an "extension penalty" to avoid buying at the peak of a price move.
       - Apply a time-decay exit for the final 30 seconds.
    """
    # Exit if too little time remains
    if seconds_remaining < 30:
        return None, 0.0

    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # Extract engineered features
    alignment_score = features_row.get('alignment_rolling', 0.5)
    volatility = features_row.get('mean_volatility', 0.0)
    price_dist = features_row.get('price_vs_strike_pct', 0.0)

    # --- 1. DYNAMIC HURDLE CALCULATION ---
    # Start with a baseline expected value requirement
    hurdle = 0.03

    # Penalty for low signal stability (flickering consensus)
    if alignment_score < 0.65:
        hurdle += 0.03
    
    # Penalty for high volatility regimes
    if volatility > 0.0015:
        hurdle += 0.02

    # --- 2. EXECUTION ---
    if best_ev > hurdle:
        # Base bet size is driven by how stable the signal is
        # If alignment is 1.0, we bet 80% of standard size. If 0.5, we bet 40%.
        bet_fraction = alignment_score * 0.8

        # Extension Penalty: Scale down if the price has already moved significantly from the strike
        # This prevents "chasing" the candle.
        extension_penalty = 1.0 / (1.0 + abs(price_dist) * 30)
        
        final_bet = min(1.0, bet_fraction * extension_penalty)
        
        # Safety floor for high-conviction trades
        if final_bet < 0.05 and best_ev > (hurdle + 0.05):
            final_bet = 0.05

        return side, final_bet

    return None, 0.0