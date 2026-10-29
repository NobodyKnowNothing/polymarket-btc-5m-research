# Var 159 | BT: -0.595 | Live PnL: $-5.25 | Trades: 60
# Desc: Implements "Congruent Flow-Model Alignment (CFMA)," which uses a dual-hurdle system that rewards trades where the model's prediction aligns with the net order flow (Delta + OBI) and heavily penalizes 
# Stats: {"live_returns": [-0.7314661741256714, 0.6997278332710266, -0.5327768325805664, 0.5144302845001221, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 1.4562249999999999, 1.9791409999999998, 0.7327178120613098, 1.9791409999999998, -1.0, 1.1846420000000002, 2.073133, -0.4750261902809143, -1.0, -0.706832230091095, -1.0, -1.0, 1.8077519999999998, -1.0, -0.872744619846344, -1.0, -1.0, -0.43755215406417847, 0.8321430087089539, -1.0, 1.3964720000000002, -1.0, -1.0, -1.0, -0.8331693410873413, -1.0, 0.12663000000000002, 1.28551, -1.0, -1.0, 0.06583833694458008, 0.108329176902771, -1.0, -1.0, -1.0, 1.890848, 0.029727280139923096, -1.0, 3.1194439999999997, -1.0, -1.0, -0.7983030676841736, -1.0, -0.5920062065124512, 0.03578603267669678, -0.7703202962875366, -1.0, 1.4562249999999999, -0.5963438749313354, 4.245841], "live_pnl": -5.253705024719238, "live_wins": 22, "live_total": 60, "age": 88, "rounds_underwater": 34, "abstain_rounds": 0, "backtest_score": -0.5949475592818287, "backtest_acc": 0.5590551181102362, "backtest_n": 3744}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the alignment between aggressive flow (Delta) and book intent (OBI).
    """
    df = df.copy()

    # 1. Squashed Delta: Bring aggressive flow into a [-1, 1] range to match OBI scale.
    # This prevents large delta spikes from overwhelming the OBI signal in the model.
    df['delta_squashed'] = np.tanh(df['mean_delta'])

    # 2. Net Flow Signal: Combined measure of intent and aggression.
    # Range: [-2, 2]. Positive = bullish flow/intent, Negative = bearish.
    df['flow_signal'] = df['delta_squashed'] + df['mean_obi']

    # 3. OBI Dispersion: Measure of how much the three exchanges disagree on the book.
    # High dispersion suggests a fragmented or highly uncertain market.
    obi_cols = ['binance_obi', 'coinbase_obi', 'kraken_obi']
    df['obi_dispersion'] = df[obi_cols].std(axis=1)

    # 4. Flow Acceleration: The rate of change of the net flow signal.
    # Captures whether order flow is ramping up or exhausting.
    df['flow_accel'] = df.groupby('market_slug')['flow_signal'].diff().fillna(0)

    # Drop metadata and target as per instructions
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Congruent Flow-Model Alignment (CFMA)".

    Logic:
    1. Calculate Congruence: 
       Does the model's predicted direction (prob_up vs 0.5) match the 
       microstructure direction (sign of flow_signal)?
    2. Dual-Hurdle Execution:
       - If Congruent: Use a low hurdle (2.5%) to capture high-conviction trends.
       - If Incongruent: Use a high hurdle (5.5%) to avoid "tug-of-war" markets.
    3. Dynamic Sizing:
       Scale bet size by:
       - The magnitude of the flow signal (strength of consensus).
       - The inverse of OBI dispersion (certainty of exchange agreement).
       - The inverse of volatility (risk management).
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    best_ev = max(ev_up, ev_down)
    
    if best_ev <= 0:
        return None, 0.0

    side = 'Up' if ev_up > ev_down else 'Down'

    # Extract engineered features
    flow_signal = features_row.get('flow_signal', 0.0)
    obi_dispersion = features_row.get('obi_dispersion', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    # 1. Determine Congruence
    model_dir = 1 if prob_up > 0.5 else -1
    flow_dir = 1 if flow_signal > 0 else -1
    is_congruent = (model_dir == flow_dir)

    # 2. Set Dynamic Hurdle
    # Congruent trades are safer; divergent trades require much higher edge.
    base_hurdle = 0.025 if is_congruent else 0.055
    # Add a small volatility buffer
    total_hurdle = base_hurdle + (mean_vol * 0.02)

    # 3. Decision and Sizing
    if best_ev > total_hurdle:
        # Flow strength: normalized [0, 1] based on the [-2, 2] range
        flow_strength = min(1.0, abs(flow_signal) / 2.0)
        
        # Confidence scaling factors
        # Penalize high dispersion (exchange disagreement) and high volatility
        uncertainty_penalty = 1.0 / (1.0 + obi_dispersion)
        vol_penalty = 1.0 / (1.0 + mean_vol)
        
        # Calculate bet fraction
        # Base edge ratio * flow strength * uncertainty/vol penalties
        edge_ratio = best_ev / total_hurdle
        bet_fraction = edge_ratio * (0.5 + 0.5 * flow_strength) * uncertainty_penalty * vol_penalty

        # Clamp bet between 0.05 and 1.0
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0