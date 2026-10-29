# Var 63 | BT: -1.084 | Live PnL: $-7.52 | Trades: 61
# Desc: Implements "Microstructure Efficiency & Funding Divergence (MEFD)," which targets high-alpha windows by identifying "Squeeze" regimes (funding/delta divergence) and "Momentum" regimes (high price-to-d
# Stats: {"live_returns": [-0.9356440901756287, -0.6372935771942139, -0.21729730069637299, 2.651597, -1.0, -0.6636483073234558, 0.6922392845153809, -0.9517138600349426, -0.698236882686615, -0.41502171754837036, 0.036707520484924316, -1.0, -0.8009835481643677, 0.05934947729110718, -0.7702142596244812, -0.5997881889343262, -1.0, 0.13749298453330994, -0.22892461717128754, -0.6269044280052185, -0.24620698392391205, 0.009085893630981445, 0.7553789019584656, 0.5519101619720459, 0.11392825841903687, -1.0, 0.009059801697731018, -0.47966012358665466, 4.550646, 2.2371912002563477, -0.6032375693321228, -1.0, 0.02151036262512207, -1.0, 0.009052947163581848, -0.27842119336128235, 0.01157689094543457, -0.5142634510993958, 2.39541, 0.5766888856887817, -0.5138535499572754, -0.8175010085105896, 0.5715683698654175, -1.0, -0.5121614933013916, -0.9497995376586914, -0.5610029697418213, -1.0, 1.7836271524429321, -0.5642345547676086, -1.0, -0.19577078521251678, 0.02890375256538391, -1.0, -1.0, 0.028974145650863647, 0.023515939712524414, -1.0, 1.2070131301879883, 0.028850555419921875, -0.22496001422405243], "live_pnl": -7.5154643058776855, "live_wins": 24, "live_total": 61, "age": 96, "rounds_underwater": 57, "abstain_rounds": 0, "backtest_score": -1.0844287628839593, "backtest_acc": 0.5546385885031303, "backtest_n": 2675}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between aggressive flow (delta) and price response (efficiency).
    """
    df = df.copy()

    # 1. Price and Delta Dynamics
    # We group by market_slug to ensure diffs are calculated within the same window
    group = df.groupby('market_slug')
    df['binance_price_change'] = group['binance_price'].diff().fillna(0)
    df['binance_delta_change'] = group['binance_delta'].diff().fillna(0)

    # 2. Price-Delta Efficiency
    # High efficiency = price moves easily with delta (Momentum)
    # Low efficiency = price resists delta (Absorption)
    # We use abs(delta) to normalize the efficiency metric
    df['price_delta_efficiency'] = df['binance_price_change'] / (df['binance_delta'].abs() + 1e-9)

    # 3. Funding-Delta Divergence (Squeeze Signal)
    # If funding is positive (longs paying) and delta is negative (selling), 
    # the product is negative, indicating a potential "long squeeze".
    df['funding_delta_divergence'] = df['mean_funding'] * df['binance_delta']

    # 4. OBI-Delta Confluence (Flow Alignment)
    # High value means OBI and aggressive flow are moving in the same direction
    df['obi_delta_confluence'] = df['mean_obi'] * df['binance_delta']

    # 5. Cross-Exchange Price Dispersion Velocity
    df['price_spread_vel'] = group['price_spread'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 12.0,
        'reg_lambda': 15.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Efficiency & Funding Divergence (MEFD)".
    
    The strategy gates trades based on three microstructure regimes:
    1. Squeeze Regime: High divergence between funding and delta (Lowers hurdle).
    2. Momentum Regime: High price-to-delta efficiency (Lowers hurdle).
    3. Absorption Regime: Low price-to-delta efficiency (Raises hurdle).
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

    # 2. Extract Microstructure Signals
    efficiency = features_row.get('price_delta_efficiency', 0.0)
    divergence = features_row.get('funding_delta_divergence', 0.0)
    obi_delta = features_row.get('obi_delta_confluence', 0.0)
    
    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 3.5% to ensure significant edge
    hurdle = 0.035

    # Signal A: Squeeze Detection (High divergence between funding and delta)
    # If funding and delta are fighting, a squeeze is likely.
    if abs(divergence) > 0.001:
        hurdle *= 0.75

    # Signal B: Absorption Detection (Price is not moving despite aggressive delta)
    # If efficiency is near zero, the market is absorbing the flow; avoid the trend.
    if abs(efficiency) < 0.0001:
        hurdle *= 1.8

    # Signal C: Flow Confluence (OBI and Delta are aligned)
    # If OBI and Delta agree, the trend is "cleaner".
    if abs(obi_delta) > 0.05:
        hurdle *= 0.85

    # 4. Execution Decision
    if best_ev < hurdle:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale by edge strength relative to the hurdle
    edge_ratio = best_ev / hurdle

    # Scale by time remaining (reduce exposure as the window closes)
    time_factor = max(0.2, seconds_remaining / 300.0)

    # Final bet fraction calculation
    # We use a conservative multiplier (0.4) to maintain high Sharpe
    bet_fraction = edge_ratio * time_factor * 0.4
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold to avoid tiny, insignificant trades
    if bet_fraction < 0.15:
        return None, 0.0

    return side, bet_fraction