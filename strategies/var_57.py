# Var 57 | BT: -0.227 | Live PnL: $-11.69 | Trades: 61
# Desc: Implements "Microstructure Coherence & Volatility-Normalized Flow (MCVF)," which scales the required edge hurdle based on the directional consensus between Order Book Imbalance, Delta, and Momentum, w
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -0.49597200751304626, -0.6192921996116638, 1.8519418239593506, -0.3406335413455963, -1.0, -0.3671470582485199, -1.0, -1.0, 1.9791409999999998, -1.0, -0.3897903561592102, -1.0, -0.68748539686203, 0.4562969207763672, -1.0, -1.0, -1.0, -0.6481047868728638, -1.0, -0.7174601554870605, -1.0, -1.0, 3.1194439999999997, -1.0, -1.0, -0.5335704684257507, 0.08768236637115479, -0.65644371509552, 2.39541, 1.28551, -0.6559205055236816, -1.0, 0.6931430101394653, -0.47929081320762634, -1.0, -0.6537606716156006, -1.0, -1.0, -1.0, 1.9791409999999998, -0.4759880006313324, 0.10282188653945923, 0.1060749888420105, -1.0, -1.0, -1.0, -1.0, 2.073133, 0.066120445728302, 3.505625, -1.0, -1.0, 1.7165616750717163, 2.795066, -0.9910467267036438], "live_pnl": -11.6910400390625, "live_wins": 17, "live_total": 61, "age": 119, "rounds_underwater": 61, "abstain_rounds": 0, "backtest_score": -0.22687716108753211, "backtest_acc": 0.5642090688819661, "backtest_n": 2021}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    volatility-normalized flow and directional alignment.
    """
    df = df.copy()

    # 1. Volatility-Normalized Flow Ratios: Is the flow significant relative to noise?
    # These help the model distinguish between high-conviction moves and random noise.
    df['delta_vol_ratio'] = df['sum_delta'] / (df['mean_volatility'] + 1e-6)
    df['obi_vol_ratio'] = df['mean_obi'] / (df['mean_volatility'] + 1e-6)
    df['mom_vol_ratio'] = df['mean_momentum'] / (df['mean_volatility'] + 1e-6)

    # 2. Price-Delta Efficiency: How much price movement is supported by aggressive flow?
    # High efficiency = price is moving cleanly with delta.
    df['price_delta_efficiency'] = df['sum_delta'] / (df['price_std'] + 1e-6)

    # 3. Funding-Flow Alignment: Does aggressive flow align with funding pressure?
    # Helps identify if flow is part of a trend or a reversal.
    df['funding_flow_alignment'] = df['sum_funding'] * df['sum_delta']

    # Drop metadata but keep all pre-parsed features and our new engineered ones
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup: handle NaNs and Infs
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 15.0,
        'reg_lambda': 20.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Microstructure Coherence Gating (MCG)".
    The required edge (hurdle) is inversely proportional to the 'coherence' 
    of the market: if OBI, Delta, and Momentum all agree, the hurdle is low. 
    If they conflict, the hurdle is very high.
    """
    # 1. Calculate raw Expected Value (EV)
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side = 'Up'
        best_ev = ev_up
    else:
        side = 'Down'
        best_ev = ev_down

    # 2. Calculate Microstructure Coherence
    # We look at the directional agreement between three key signals.
    def get_sign(val):
        if val > 0: return 1
        if val < 0: return -1
        return 0

    s_delta = get_sign(features_row.get('sum_delta', 0.0))
    s_obi = get_sign(features_row.get('mean_obi', 0.0))
    s_mom = get_sign(features_row.get('mean_momentum', 0.0))

    # Coherence score ranges from 0.0 (total disagreement) to 1.0 (perfect alignment)
    # Using abs(sum) / 3 captures the magnitude of consensus.
    coherence = abs(s_delta + s_obi + s_mom) / 3.0

    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 4%. If coherence is 1.0, hurdle is ~3.2%.
    # If coherence is 0.0, hurdle jumps to 16% (protecting against noise).
    hurdle = 0.04 / (coherence + 0.25)

    # 4. Execution Decision
    if best_ev < hurdle:
        return None, 0.0

    # 5. Dynamic Sizing
    # Scale bet size by:
    # - The ratio of actual edge to required hurdle
    # - Time decay: reduce size as the window closes
    time_factor = max(0.0, seconds_remaining / 300.0)
    edge_strength = best_ev / hurdle

    bet_fraction = edge_strength * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum bet threshold
    if bet_fraction < 0.15:
        return None, 0.0

    return side, bet_fraction