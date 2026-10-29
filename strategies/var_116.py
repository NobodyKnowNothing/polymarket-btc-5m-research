# Var 116 | BT: 0.378 | Live PnL: $-1.31 | Trades: 168
# Desc: Implements "Coherence-Weighted Edge Capture (CWEC)," which scales the expected value and bet size by a microstructure coherence factor that combines cross-exchange price consensus with the directional
# Stats: {"live_returns": [2.795066, 1.1046732664108276, 1.1046319007873535, -1.0, -1.0, -0.9363054633140564, 0.15118699999999996, 0.246969, 1.233902, 2.8185410499572754, 1.1846420000000002, -1.0, -1.0, -0.42076390981674194, -1.0, 3.9731249999999996, -1.0, -1.0, -1.0, 3.505625, 0.5773733258247375, 1.5137156248092651, -0.6072009205818176, -0.5254730582237244, -1.0, 4.893558, -1.0, -0.5431591868400574, -1.0, 6.238865, 3.727685, -1.0, -1.0, -1.0, 3.1194439999999997, -0.6289536952972412, -1.0, -1.0, -1.0, -1.0, -1.0, 2.950496, 0.6005876064300537, 3.9731249999999996, -1.0, -1.0, 0.783041775226593, 0.10316799999999993, 2.280573, 1.1143531799316406, -0.9868050813674927, -1.0, 1.8692638874053955, -1.0, -0.946162223815918, -1.0, -1.0, -0.3563929796218872, -1.0, 2.950496, -1.0, 0.8877313137054443, -1.0, -0.5480873584747314, -1.0, -1.0, -0.7305775880813599, -0.5842469334602356, -1.0, -1.0, -1.0, 0.6135689616203308, -1.0, -1.0, -1.0, 1.2303433418273926, 2.651597, -1.0, -1.0, -1.0, -1.0, -0.683976948261261, -0.5616004467010498, -0.8791528940200806, -1.0, 1.1829100847244263, -0.5616981983184814, -1.0, 4.550646, -1.0, -1.0, -1.0, 1.3396370000000002, -1.0, -1.0, -1.0, -1.0, -1.0, 0.7958341836929321, -1.0, -1.0, -1.0, -1.0, 3.727685, 2.795066, 1.9791409999999998, -1.0, 1.28551, 2.4428577423095703, 3.1194439999999997, -0.46273916959762573, 1.9791409999999998, -0.7501753568649292, -1.0, -0.41256341338157654, 3.303756, -1.0, 0.06987699999999997, -1.0, -1.0, -1.0, -0.624329149723053, -1.0, -1.0, 0.11476700000000006, 2.795066, -0.49132516980171204, -1.0, 1.7985994815826416, -0.6652254462242126, -1.0, -1.0, -1.0, -0.5016844272613525, -1.0, -1.0, -1.0, 0.9636929631233215, 0.038675000000000015, -1.0, -1.0, 1.4497108459472656, 1.7823529243469238, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.795066, -0.6405060887336731, -0.3953739404678345, 0.0627211332321167, -1.0, -1.0, -1.0, -1.0, 0.7442953586578369, -1.0, -1.0, 0.09182599999999996, -1.0, 1.7294070000000001, -1.0, -1.0, -1.0, -1.0], "live_pnl": -1.3094429969787598, "live_wins": 54, "live_total": 168, "age": 308, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.37783592622468243, "backtest_acc": 0.5606557377049181, "backtest_n": 3230}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    scale-invariant price consensus and the derivatives of order flow.
    """
    df = df.copy()

    # 1. Scale-Invariant Price Consensus
    # Measures how much exchanges agree on the price relative to the price level.
    # Higher = more consensus.
    df['price_consensus'] = 1.0 / (1.0 + (df['price_spread'] / (df['binance_price'] + 1e-6)))

    # 2. Order Flow Dynamics
    # First and second derivatives of the mean delta to capture flow velocity and acceleration.
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['delta_acceleration'] = df.groupby('market_slug')['mean_delta'].diff().diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle inf and NaNs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Coherence-Weighted Edge Capture (CWEC)".

    Logic:
    - Instead of a binary gate, we calculate a 'microstructure coherence' score.
    - Coherence = (Price Consensus) * (Directional Alignment of OBI and Delta).
    - The model's Expected Value (EV) is scaled by this coherence.
    - If OBI and Delta are fighting (divergence), coherence drops significantly, 
      effectively neutralizing the edge and reducing bet size.
    - If OBI and Delta are aligned (congruence), coherence increases, 
      amplifying the signal and bet size.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    consensus = features_row.get('price_consensus', 0.5)
    obi = features_row.get('mean_obi', 0.0)
    delta = features_row.get('mean_delta', 0.0)

    # 1. Calculate Microstructure Coherence
    # Base coherence is the price consensus (how much exchanges agree).
    coherence = consensus

    # Adjust coherence based on OBI and Delta alignment.
    # If they move in the same direction, it's high-conviction (boost coherence).
    # If they move in opposite directions, it's absorption/noise (penalize coherence).
    if (obi > 0 and delta > 0) or (obi < 0 and delta < 0):
        # Boost coherence by the strength of the alignment
        coherence *= (1.0 + abs(obi * delta))
    else:
        # Heavy penalty for divergence (potential fakeout/absorption)
        coherence *= 0.4

    # 2. Apply Coherence to EV
    # We treat the model's edge as being "diluted" by market noise.
    effective_ev = best_ev * coherence

    # 3. Time-Gated Hurdle
    # As the window closes, increase the required edge to account for late-stage volatility.
    time_factor = 1.0 + (60.0 / (seconds_remaining + 1.0))
    threshold = 0.025 * time_factor

    if effective_ev > threshold:
        # 4. Dynamic Sizing
        # Scale bet size by the effective edge relative to a 4% target, 
        # further dampened by the coherence factor.
        bet_fraction = (effective_ev / 0.04) * coherence
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0