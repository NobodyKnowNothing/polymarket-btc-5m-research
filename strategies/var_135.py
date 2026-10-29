# Var 135 | BT: -0.467 | Live PnL: $-7.62 | Trades: 61
# Desc: Implements "Impulse-Coherence Gating (ICG)," which filters for high-conviction trades where aggressive delta, order book imbalance, and price movement all align in a "clean" impulse relative to curren
# Stats: {"live_returns": [-1.0, 0.09182599999999996, -1.0, -1.0, -0.5380164980888367, 0.09182599999999996, -1.0, 1.3708970546722412, -0.7926976680755615, -1.0, -1.0, -0.6015539169311523, -0.7500335574150085, -0.432808518409729, 2.39541, 1.1082355976104736, 1.7319347858428955, -0.9233057498931885, 1.890848, -1.0, -1.0, 0.8113253712654114, 0.04130834341049194, 1.4562249999999999, 1.890848, 1.7294070000000001, 1.28551, -0.45364370942115784, 3.727685, -1.0, -0.3712080419063568, 0.3798684775829315, -1.0, 2.39541, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4222765862941742, -1.0, -0.8140642642974854, -1.0, -1.0, -1.0, 0.09966528415679932, 1.541098713874817, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5013372898101807, -1.0, 0.10316799999999993, 1.1846420000000002, 1.8077519999999998, -1.0, -1.0, -1.0], "live_pnl": -7.621501922607422, "live_wins": 22, "live_total": 61, "age": 114, "rounds_underwater": 18, "abstain_rounds": 0, "backtest_score": -0.4670650043221264, "backtest_acc": 0.5732905554441547, "backtest_n": 3028}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw features into impulse-based features that measure the 
    quality of price movement relative to volatility.
    """
    df = df.copy()

    # 1. Impulse Features: Measuring the magnitude of flow/imbalance per unit of volatility
    # This helps distinguish between "noisy" high-volatility moves and "clean" low-volatility trends.
    vol_adj = df['mean_volatility'] + 1e-6
    
    df['delta_impulse'] = df['sum_delta'] / vol_adj
    df['obi_impulse'] = df['mean_obi'] / vol_adj
    df['price_impulse'] = (df['binance_price'] - df['strike_price']) / vol_adj

    # 2. Coherence Score: How many signals (Delta, OBI, Price) are pointing in the same direction?
    # We use np.sign and ensure we only count matches where the delta is non-zero.
    s_delta = np.sign(df['delta_impulse'])
    s_obi = np.sign(df['obi_impulse'])
    s_price = np.sign(df['price_impulse'])

    # Score is 0, 1, or 2. 2 = Perfect alignment (Delta, OBI, and Price all agree).
    # We only count a match if delta is actually present (s_delta != 0).
    df['coherence_score'] = (
        ((s_delta != 0) & (s_delta == s_obi)).astype(int) + 
        ((s_delta != 0) & (s_delta == s_price)).astype(int)
    )

    # 3. Efficiency: Price movement per unit of delta (detecting absorption vs sweeps)
    df['price_delta_efficiency'] = (df['binance_price'] - df['strike_price']) / (df['sum_delta'].abs() + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on strong impulse signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 25.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Impulse-Coherence Gating (ICG)".

    Logic:
    - Calculate raw EV for both sides.
    - Coherence Gating: We look at the 'coherence_score' (0, 1, or 2).
    - If coherence is 0 (no alignment between Delta, OBI, and Price), we abstain.
    - If coherence is 1 (partial alignment), we require a higher EV threshold (0.04).
    - If coherence is 2 (perfect alignment), we lower the threshold (0.02) and 
      increase the bet size.
    - This targets "clean" momentum moves and avoids "noisy" or "absorbed" markets.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    coherence = features_row.get('coherence_score', 0.0)

    # 1. Absolute Abstention on no coherence
    if coherence < 1:
        return None, 0.0

    # 2. Dynamic Thresholding based on Coherence
    # Perfect alignment (2) allows for smaller edges; partial alignment (1) requires more.
    if coherence >= 2:
        threshold = 0.02
        bet_multiplier = 1.0
    else:
        threshold = 0.04
        bet_multiplier = 0.5

    # 3. Time-Decay Adjustment
    # As we approach the end, we become more conservative with the threshold.
    if seconds_remaining < 45:
        threshold += 0.01

    if best_ev > threshold:
        # Scale bet fraction by EV strength and coherence strength
        # A perfect coherence (2) with 4% EV gives a 1.0 bet.
        # A partial coherence (1) with 4% EV gives a 0.5 bet.
        bet_fraction = (best_ev / 0.04) * (coherence / 2.0) * bet_multiplier
        return side, min(1.0, max(0.1, bet_fraction))

    return None, 0.0