# Var 17 | BT: -0.338 | Live PnL: $-22.40 | Trades: 61
# Desc: Implements "Binance-Lead Arbitrage & Alignment Capture," which treats Binance as the primary price discovery engine and prioritizes trades where Binance's price lead and delta flow are congruent, sign
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 1.5191249999999998, 3.1194439999999997, 1.6554160000000002, -1.0, -1.0, -0.6546617746353149, -1.0, -0.8189799785614014, 1.890848, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9862099885940552, -0.7072193622589111, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.550646, -1.0, 3.1194439999999997, -1.0, -0.678333580493927, -1.0, -1.0, -1.0, -1.0, -0.9618576765060425, -1.0, -1.0, -0.6341425776481628, 2.39541, 0.05437976121902466, 1.1638529300689697, -1.0, -1.0, 1.008033037185669, -0.7223470211029053, -1.0, -1.0, -1.0, -0.6446092128753662, -1.0, -1.0, 1.9791409999999998, -1.0, -1.0, -0.5635905265808105], "live_pnl": -22.397085189819336, "live_wins": 12, "live_total": 61, "age": 73, "rounds_underwater": 46, "abstain_rounds": 0, "backtest_score": -0.3379050559109469, "backtest_acc": 0.5370675453047776, "backtest_n": 1969}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Binance price leadership and delta-OBI congruence.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Binance Lead-Lag Features
    # Calculate the average price across all three exchanges
    avg_price = (X['binance_price'] + X['coinbase_price'] + X['kraken_price']) / 3.0
    X['binance_price_lead'] = X['binance_price'] - avg_price
    
    # Calculate how much Binance delta deviates from the mean delta
    X['binance_delta_lead'] = X['binance_delta'] - X['mean_delta']
    
    # Alignment: 1 if price and delta lead are in same direction, -1 if opposite, 0 if neutral
    X['lead_alignment'] = np.sign(X['binance_price_lead']) * np.sign(X['binance_delta_lead'])
    
    # Strength: How many standard deviations the Binance lead is from the cross-exchange dispersion
    X['price_lead_strength'] = X['binance_price_lead'].abs() / (X['price_std'] + 1e-7)

    # 2. Microstructure Congruence
    # Measures if aggressive flow (delta) is aligned with the order book imbalance (OBI)
    X['delta_obi_congruence'] = np.sign(X['mean_delta']) * X['mean_obi']

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to focus on lead-lag signals."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using a "Lead-Lag" logic.
    We seek moments where Binance is leading the price move with aggressive delta,
    suggesting the Polymarket cost is lagging behind the true market direction.
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

    # 2. Extract Lead-Lag Metrics
    # lead_alignment: 1 (congruent), -1 (divergent), 0 (neutral)
    lead_alignment = features_row.get('lead_alignment', 0)
    # price_lead_strength: magnitude of Binance's price deviation
    price_lead_strength = features_row.get('price_lead_strength', 0)

    # 3. Dynamic Hurdle Logic
    # Base hurdle is 2.5%
    base_hurdle = 0.025
    
    # Penalty for Divergence: If Binance price is moving one way but delta is moving the other, 
    # it's likely absorption or noise. Increase hurdle significantly.
    if lead_alignment <= 0:
        hurdle = base_hurdle + 0.035
    else:
        hurdle = base_hurdle

    # 4. Execution Logic
    if best_ev > hurdle:
        # Time-decay: Reduce exposure as the window closes to avoid volatility spikes
        time_factor = max(0.4, min(1.0, seconds_remaining / 300.0))

        # Confidence Scaling: Scale bet size by the strength of the Binance lead.
        # If Binance is leading by a large margin (high price_lead_strength), we bet more.
        # We use a sigmoid-like scaling to cap the influence of extreme outliers.
        confidence_multiplier = 0.5 + (0.5 * (1.0 - math.exp(-price_lead_strength)))
        
        # Bet sizing: (Edge / Hurdle) * Confidence * Time
        bet_fraction = (best_ev / hurdle) * confidence_multiplier * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold to ensure trades are worth the risk
        if bet_fraction < 0.15:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0