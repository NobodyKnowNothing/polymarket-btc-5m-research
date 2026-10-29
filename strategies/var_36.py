# Var 36 | BT: -1.305 | Live PnL: $-16.99 | Trades: 61
# Desc: Targets "Stable Momentum" by increasing the entry hurdle during periods of high cross-exchange price dispersion or extreme funding, and penalizing trades where order flow and book pressure diverge.
# Stats: {"live_returns": [0.13876599999999994, 1.890848, 0.02137303352355957, -1.0, 0.07973021268844604, -1.0, 0.009782850742340088, 0.08073200000000003, 1.6554160000000002, -0.9670297503471375, -0.7311909198760986, 0.10316799999999993, -1.0, -1.0, -1.0, -0.9014731049537659, -1.0, 0.7696995735168457, 0.23225600000000002, -1.0, -1.0, -0.30251580476760864, -1.0, -0.8481087684631348, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4647372364997864, 1.1846420000000002, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, 0.050668954849243164, -1.0, -1.0, -1.0, -1.0, -0.40883398056030273, -1.0, -1.0, -1.0, 2.39541, 0.02197858691215515, 1.28551, -0.4976275563240051, -0.9850852489471436, 0.9097610116004944, -1.0, -1.0, -0.47437620162963867, -1.0, -1.0, -1.0, -1.0, 1.9791409999999998, 0.11476700000000006, -1.0], "live_pnl": -16.98723793029785, "live_wins": 20, "live_total": 61, "age": 65, "rounds_underwater": 49, "abstain_rounds": 0, "backtest_score": -1.3047025663445961, "backtest_acc": 0.5503638452700115, "backtest_n": 2273}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    microstructure stability (spread/price ratio) and flow-book congruence.
    """
    df = df.copy()

    # 1. Congruence: Do aggressive trades (Delta) align with the book (OBI)?
    # Positive = Flow and Book are pushing in the same direction.
    # Negative = Absorption (Flow hitting a wall).
    df['obi_delta_congruence'] = df['mean_obi'] * df['mean_delta']

    # 2. Stability: How much do exchanges disagree?
    # We use the price spread relative to the absolute price.
    df['spread_to_price_ratio'] = df['price_spread'] / (df['binance_price'] + 1e-6)

    # 3. Funding-Flow Interaction: Is the trend supported by perp positioning?
    df['funding_flow_interaction'] = df['mean_funding'] * df['mean_delta']

    # 4. Velocity: Price change per snapshot
    df['binance_vel'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata but keep pm_costs and other features for the model
    # We must keep 'seconds_remaining' and 'pm_cost_up/down' for the strategy_filter, 
    # but they are not features for the XGBoost model itself.
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Stable Momentum Capture".
    Filters out trades during high-dispersion (unstable) or high-absorption (fake) regimes.
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
    congruence = features_row.get('obi_delta_congruence', 0.0)
    spread_ratio = features_row.get('spread_to_price_ratio', 0.0)
    mean_funding = features_row.get('mean_funding', 0.0)
    
    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 2.5%. 
    # We increase the hurdle if:
    # - Exchanges disagree (high spread_ratio)
    # - Funding is extreme (high abs(mean_funding)), suggesting a potential squeeze/reversal
    dynamic_hurdle = 0.025 + (5.0 * spread_ratio) + (0.5 * abs(mean_funding))

    if best_ev < dynamic_hurdle:
        return None, 0.0

    # 4. Confluence & Absorption Penalty
    # If delta and OBI are fighting (congruence < 0), the move is likely being absorbed.
    # We heavily penalize the bet size in absorption regimes.
    multiplier = 1.0
    if congruence < 0:
        multiplier = 0.25 
    elif abs(congruence) < 0.01: # Very low signal strength
        multiplier = 0.6

    # 5. Time-decay: Reduce exposure as the window closes to avoid late-window volatility
    time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

    # 6. Final Bet Sizing
    # Scale by the ratio of edge to hurdle, adjusted by confluence and time.
    bet_fraction = (best_ev / dynamic_hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Minimum threshold to ensure the trade is worth the potential spread/slippage
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction