# Var 195 | BT: -0.628 | Live PnL: $-8.71 | Trades: 60
# Desc: Implements "Microstructure-Sentiment Divergence (MSD)," which targets mispricings by identifying gaps between the "real" market sentiment (a composite of OBI and aggressive Delta) and the "prediction"
# Stats: {"live_returns": [-0.6369262581056391, 1.9791409999999998, 1.8033874682016595, 2.0302217944804086, -0.8840494797400434, -0.7609755243516815, 1.9570757310751046, 1.225119831180075, -1.0, -0.6876277085322053, -0.9404286099372223, -0.8454263059239039, -0.8338419543381411, -0.9186184671823445, -1.0, -0.6592253579651611, 1.2679218205315235, -0.7050422820589114, 1.9610935876571456, -0.5767842097472476, 1.501346801112881, -1.0, -0.862473286536327, -1.0, -0.83268987816952, -0.5901763610528802, -0.799297772940775, 1.1558419547123293, -0.7751834264643159, 1.890848, -0.8944381087181253, 1.1876981233395774, -1.0, -0.7937656350752331, -0.5991340042074379, -0.7966423777347088, -0.8186500600801286, 1.2631916575314284, -0.6141964461179354, 1.8282821075278894, -0.6901149079375966, -0.880682437995861, -1.0, 0.8596760879350092, -0.8798757796286709, -0.686012110227458, -1.0, -0.9120667717295171, -0.9489512778596346, -0.9853002653647979, 2.39541, -0.714755540287619, -1.0, -1.0, 0.7191584240684384, -0.6743869524108564, 2.173394, -0.9782130963150613, -1.0, -0.7288087720221516], "live_pnl": -8.705953037405646, "live_wins": 17, "live_total": 60, "age": 139, "rounds_underwater": 28, "abstain_rounds": 0, "backtest_score": -0.6279421618895105, "backtest_acc": 0.5711753235241979, "backtest_n": 2677}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, specifically creating 
    a 'divergence' metric between microstructure sentiment and Polymarket pricing.
    """
    df = df.copy()

    # 1. Calculate Microstructure Sentiment
    # We combine OBI (intent) and normalized Delta (action) into a single [-1, 1] signal.
    # We use price_std as a proxy to normalize Delta, ensuring it's on a similar scale to OBI.
    delta_normalized = np.tanh(df['mean_delta'] / (df['price_std'] + 1e-6))
    micro_sentiment = (df['mean_obi'] + delta_normalized) / 2
    
    # 2. Map Microstructure Sentiment to 0-1 scale to match Polymarket costs
    df['micro_sentiment_01'] = (micro_sentiment + 1) / 2

    # 3. Calculate Sentiment Divergence
    # Positive divergence: Microstructure is more bullish than Polymarket price.
    # Negative divergence: Microstructure is more bearish than Polymarket price.
    df['divergence'] = df['micro_sentiment_01'] - df['pm_cost_up']

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to divergence noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure-Sentiment Divergence (MSD)".

    Logic:
    The model provides the probability, but we only execute if the 'divergence' 
    feature confirms that the Polymarket price is lagging behind the actual 
    microstructure momentum.
    """
    # 1. Time Gate: Avoid high-slippage/low-liquidity environments near close
    if seconds_remaining < 60:
        return None, 0.0

    # 2. Calculate Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # 3. Extract divergence feature
    # divergence > 0 means Micro is Bullish, PM is Bearish
    # divergence < 0 means Micro is Bearish, PM is Bullish
    div = features_row.get('divergence', 0.0)

    side = None
    bet_fraction = 0.0

    # 4. Execution Logic: Gated by Divergence Confirmation
    # We require a minimum EV and a minimum divergence to ensure the 'gap' is significant.
    
    # BULLISH CASE: Model sees edge AND microstructure is significantly more bullish than PM
    if ev_up > 0.025 and div > 0.08:
        side = 'Up'
        # Scale bet size: higher divergence = higher conviction
        bet_fraction = min(1.0, 0.4 + (div * 2.0))

    # BEARISH CASE: Model sees edge AND microstructure is significantly more bearish than PM
    elif ev_down > 0.025 and div < -0.08:
        side = 'Down'
        # Scale bet size: higher absolute divergence = higher conviction
        bet_fraction = min(1.0, 0.4 + (abs(div) * 2.0))

    # Final safety clamp
    if side:
        return side, max(0.1, min(1.0, bet_fraction))

    return None, 0.0