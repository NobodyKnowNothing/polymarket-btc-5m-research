# Var 150 | BT: -0.158 | Live PnL: $-8.60 | Trades: 60
# Desc: Implements "Signal-to-Noise Ratio (SNR) Gating," which dynamically adjusts the required Expected Value (EV) threshold based on the ratio of order flow coherence to market dispersion and volatility.
# Stats: {"live_returns": [-0.5432989597320557, -0.05789351090788841, -1.0, 2.073133, -0.41397544741630554, 1.1175167560577393, -0.16007311642169952, -1.0, -0.05, 0.9649516344070435, -0.1579398661851883, -0.7243826985359192, -1.0, -0.5772979855537415, -0.2794879972934723, -0.05, -0.07124394178390503, -1.0, -1.0, 3.1194439999999997, -0.8578822016716003, -1.0, -1.0, -1.0, 0.0088023841381073, 0.5034216642379761, -0.1068907082080841, -1.0, -0.15801163017749786, -0.1822584569454193, -0.30247700214385986, -1.0, 1.5191249999999998, 2.0252583026885986, -0.5094612240791321, 0.09751027822494507, -1.0, 0.5373850464820862, -1.0, -1.0, -0.7493791580200195, -0.4298335909843445, -1.0, 0.6756101846694946, -0.1442452222108841, 1.0511279106140137, 1.9791409999999998, 0.236594095826149, -0.05, 1.1846420000000002, 2.073133, -1.0, -0.8922557830810547, -1.0, -0.9197712540626526, -1.0, 0.3359018564224243, -1.0, -1.0, -0.7185638546943665], "live_pnl": -8.603925704956055, "live_wins": 17, "live_total": 60, "age": 118, "rounds_underwater": 48, "abstain_rounds": 0, "backtest_score": -0.15785953704352124, "backtest_acc": 0.56601607347876, "backtest_n": 3042}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the signal-to-noise ratio of the microstructure.
    """
    df = df.copy()

    # 1. Coherence: Do the intent (OBI) and the action (Delta) align in magnitude?
    # We use absolute value for the quality metric, but the model sees the raw product.
    df['coherence'] = df['mean_obi'] * df['mean_delta']
    df['abs_coherence'] = df['coherence'].abs()

    # 2. Market Noise: Combined measure of price dispersion, book imbalance, and volatility.
    # We add a small constant to prevent division by zero and dampen extreme noise.
    df['market_noise'] = (df['price_spread'] + df['obi_spread']) * (df['mean_volatility'] + 0.05) + 1e-6

    # 3. Signal Quality: The ratio of coherent flow to market noise.
    # High SNR means the flow is strong relative to the current market chaos.
    df['signal_quality'] = df['abs_coherence'] / df['market_noise']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return robust hyperparameters with heavy regularization to combat noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 15.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Signal-to-Noise Ratio (SNR) Gating".

    Logic:
    1. Identify the best EV side.
    2. Calculate a dynamic EV threshold: 
       - In high-quality environments (high SNR), we accept lower EV.
       - In noisy environments (low SNR), we require a much higher EV to trade.
    3. Scale bet size by both the magnitude of the edge and the signal quality.
    4. Apply time-decay to reduce exposure as the window approaches zero.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side, ev = 'Up', ev_up
    else:
        side, ev = 'Down', ev_down

    # Extract engineered quality metric
    quality = features_row.get('signal_quality', 0.0)

    # Dynamic Thresholding: threshold = 0.05 / (quality + 0.5)
    # If quality is 2.0 (high), threshold is ~0.02 (lenient)
    # If quality is 0.1 (low), threshold is ~0.08 (strict)
    required_ev = 0.05 / (quality + 0.5)

    if ev > required_ev:
        # Base bet scales with edge (target 4% EV for full bet)
        # Multiplied by quality to favor high-conviction signals
        bet_fraction = (ev / 0.04) * min(1.0, quality)

        # Time-decay factor: reduce bet size if we are in the final 60 seconds
        # to avoid being caught in end-of-window volatility spikes.
        time_factor = 1.0
        if seconds_remaining < 60:
            time_factor = seconds_remaining / 60.0

        final_bet = bet_fraction * time_factor
        
        # Clamp bet between 0.05 and 1.0
        return side, min(1.0, max(0.05, final_bet))

    return None, 0.0