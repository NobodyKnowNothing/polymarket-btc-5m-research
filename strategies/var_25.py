# Var 25 | BT: -0.118 | Live PnL: $-8.07 | Trades: 60
# Desc: Implements "Lead-Exchange Conviction & Cross-Delta Consensus (LECDC)," which uses the magnitude of the single most aggressive exchange's order-flow-imbalance product to scale bet sizes, while using cr
# Stats: {"live_returns": [-1.0, -1.0, -0.785590648651123, -1.0, 0.7237808704376221, 3.1194439999999997, 0.07004541158676147, -1.0, -1.0, -1.0, -1.0, 0.10316799999999993, -1.0, 2.651597, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8034663200378418, -1.0, -1.0, 1.5191249999999998, 2.651597, -1.0, -1.0, 1.3396370000000002, 4.893558, -1.0, -0.6593208909034729, -1.0, -1.0, -1.0, -1.0, 2.073133, -1.0, 0.09433084726333618, 1.4562249999999999, 1.9791409999999998, 1.9791409999999998, -1.0, 1.1846420000000002, 2.073133, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.8077519999999998, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0666455030441284, -1.0, 1.3964720000000002, -1.0, -1.0], "live_pnl": -8.065811157226562, "live_wins": 19, "live_total": 60, "age": 95, "rounds_underwater": 16, "abstain_rounds": 0, "backtest_score": -0.11838524816533044, "backtest_acc": 0.531563981042654, "backtest_n": 3780}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    lead-exchange conviction and cross-exchange delta agreement.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Delta Agreement: Measures how much the three exchanges' aggressive flows agree.
    # Uses the range of deltas; 1.0 is perfect agreement, approaches 0.0 as disagreement grows.
    deltas = df[['binance_delta', 'coinbase_delta', 'kraken_delta']]
    delta_range = deltas.max(axis=1) - deltas.min(axis=1)
    X['delta_agreement'] = 1.0 / (1.0 + delta_range)

    # 2. Lead Conviction Magnitude: The absolute strength of the most aggressive exchange.
    # Calculated as |Delta * OBI| for each exchange, taking the maximum.
    p1 = (df['binance_delta'] * df['binance_obi']).abs()
    p2 = (df['coinbase_delta'] * df['coinbase_obi']).abs()
    p3 = (df['kraken_delta'] * df['kraken_obi']).abs()
    X['lead_conviction_mag'] = np.maximum(p1, np.maximum(p2, p3))

    # 3. Funding-Volatility Interaction: Captures the impact of funding in high-vol regimes.
    X['funding_vol_interaction'] = df['mean_funding'] * df['mean_volatility']

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prevent overfitting to noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 25.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Lead-Exchange Conviction and Delta Consensus.
    The hurdle scales up with disagreement, while the bet size scales up with 
    the conviction of the leading exchange.
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

    # 2. Extract engineered metrics
    agreement = features_row.get('delta_agreement', 0.5)
    conviction_mag = features_row.get('lead_conviction_mag', 0.0)
    mean_vol = features_row.get('mean_volatility', 0.0)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle is 2.5%. 
    # Penalty for low delta agreement (fragmented flow).
    # Penalty for high volatility (increased noise).
    agreement_penalty = 0.02 * (1.0 - agreement)
    vol_penalty = 0.015 if mean_vol > 0.002 else 0.0
    
    dynamic_hurdle = 0.025 + agreement_penalty + vol_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Time-decay: Reduce bet size as we approach the 5-min resolution.
        time_factor = max(0.3, min(1.0, seconds_remaining / 300.0))

        # Bet sizing:
        # Scale linearly with edge/hurdle ratio.
        # Boost bet size based on the magnitude of the leading exchange's conviction.
        base_bet = (best_ev / dynamic_hurdle)
        conviction_boost = 1.0 + conviction_mag
        
        bet_fraction = base_bet * time_factor * conviction_boost
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure trades are meaningful
        if bet_fraction < 0.15:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0