# Var 161 | BT: 0.862 | Live PnL: $-11.59 | Trades: 60
# Desc: Implements "Consensus-Validated Odds Divergence (CVOD)," which targets the discrepancy between model-predicted probabilities and market prices, but applies a heavy penalty to the required Expected Val
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 2.069054126739502, -1.0, 1.8153743743896484, 1.9791409999999998, 1.8240891695022583, -1.0, -1.0, -1.0, 1.4519155025482178, -1.0, -0.7769577503204346, -0.9059993028640747, -0.9550713300704956, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9934569597244263, 3.1194439999999997, -1.0, -0.791801929473877, -1.0, -1.0, -1.0, -0.8028467893600464, 1.4562249999999999, 4.245841, 2.518758, -1.0, -1.0, 2.518758, -1.0, -1.0, -0.8408534526824951, -1.0, -1.0, -0.9543944597244263, -1.0, -1.0, -1.0, 2.795066, -1.0, -1.0, -1.0, 9.427877, -1.0, -1.0, -1.0, -0.786766529083252, -1.0, -1.0], "live_pnl": -11.586605072021484, "live_wins": 12, "live_total": 60, "age": 144, "rounds_underwater": 46, "abstain_rounds": 0, "backtest_score": 0.8619585823696647, "backtest_acc": 0.553223388305847, "backtest_n": 2399}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    market stability metrics and flow-pressure interactions.
    """
    X = df.copy()

    # 1. Normalized Price Dispersion: How much exchanges disagree on price.
    # High dispersion = fragmented/unreliable market.
    X['price_spread_pct'] = X['price_spread'] / (X['binance_price'] + 1e-6)

    # 2. Order Book Stability: Inverse of the OBI spread.
    # High stability = tight, reliable order book intent.
    X['obi_stability'] = 1.0 / (1.0 + X['obi_spread'])

    # 3. Delta Flow Intensity: Aggressive volume relative to volatility.
    # High intensity = significant aggressive movement relative to noise.
    X['delta_flow_intensity'] = X['mean_delta'].abs() / (X['mean_volatility'] + 1e-6)

    # 4. Funding Pressure: The interaction between funding rates and aggressive flow.
    # Helps identify if aggressive trades are aligned with perp positioning.
    X['funding_pressure'] = X['mean_funding'] * X['sum_delta']

    # Drop metadata and target as per instructions
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to focus on robust signals."""
    return {
        'n_estimators': 800,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Consensus-Validated Odds Divergence (CVOD)".

    Logic:
    1. Calculate Expected Value (EV) for both sides.
    2. Calculate a "Market Integrity Hurdle":
       The required EV increases significantly if:
       - Price dispersion across exchanges is high (price_spread_pct).
       - The order book is wide/unstable (obi_spread).
       - The market window is nearly closed (time decay).
    3. Execution:
       - Only trade if the best EV exceeds the integrity-adjusted hurdle.
       - Scale bet size based on how much the EV exceeds that hurdle.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    best_ev = max(ev_up, ev_down)
    side = 'Up' if ev_up > ev_down else 'Down'

    # Extract engineered features for stability assessment
    price_spread_pct = features_row.get('price_spread_pct', 0.0)
    obi_spread = features_row.get('obi_spread', 0.0)

    # 1. Calculate Dynamic Integrity Hurdle
    # Base hurdle is 2.5%.
    # Price fragmentation adds up to 5% (at 0.12% spread).
    # OBI instability adds up to 2% (at 0.2 spread).
    # Time decay adds up to 3% as the window closes.
    base_hurdle = 0.025
    fragmentation_penalty = price_spread_pct * 40.0
    book_instability_penalty = obi_spread * 0.1
    time_decay_penalty = max(0.0, (300 - seconds_remaining) / 300 * 0.03)

    total_hurdle = base_hurdle + fragmentation_penalty + book_instability_penalty + time_decay_penalty

    # 2. Decision and Sizing
    if best_ev > total_hurdle:
        # Scale bet fraction:
        # - Higher edge relative to hurdle = larger bet.
        # - We use a conservative multiplier (0.75) to protect Sharpe.
        edge_ratio = best_ev / total_hurdle
        bet_fraction = min(1.0, edge_ratio * 0.75)
        
        return side, max(0.05, bet_fraction)

    return None, 0.0