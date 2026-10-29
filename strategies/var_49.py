# Var 49 | BT: -0.598 | Live PnL: $-22.36 | Trades: 61
# Desc: Implements "Consensus-Gated Impulse Scaling (CGIS)," which targets high-conviction microstructure impulses by requiring alignment between aggressive flow (Delta) and order book pressure (OBI) across a
# Stats: {"live_returns": [-1.0, 1.5854279999999998, 1.3964720000000002, -1.0, -0.07400473207235336, -1.0, -0.2510397136211395, -1.0, -0.2927171587944031, -0.19132177531719208, -0.6074384450912476, -1.0, -1.0, -1.0, -0.3353714644908905, -0.45916110277175903, -1.0, -0.7180362939834595, -1.0, -0.21592244505882263, -1.0, -1.0, -0.504715621471405, -0.33193138241767883, -0.5714834928512573, -1.0, -0.28150925040245056, -1.0, -0.9009163975715637, -0.4116438627243042, 0.021744370460510254, 3.1194439999999997, -1.0, -1.0, 0.08073200000000003, -0.3432192802429199, -1.0, -0.10865674912929535, 0.030405640602111816, -0.7042351961135864, 2.39541, 0.7414014935493469, -0.1614239662885666, -0.22962087392807007, 0.3302140235900879, -0.09280652552843094, -0.11262467503547668, -1.0, -0.44334450364112854, -1.0, -1.0, 0.6454415321350098, -0.2779393196105957, 0.04495379328727722, 0.04400968551635742, -1.0, -1.0, -1.0, -0.5106221437454224, -1.0, -0.6671352386474609], "live_pnl": -22.36318588256836, "live_wins": 12, "live_total": 61, "age": 84, "rounds_underwater": 56, "abstain_rounds": 0, "backtest_score": -0.5980356229593722, "backtest_acc": 0.5390455531453362, "backtest_n": 2330}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    consensus-based microstructure signals and momentum efficiency.
    """
    df = df.copy()

    # 1. OBI Consensus: How much do the exchanges agree on the book pressure?
    # Normalized so 1.0 is perfect agreement and 0.0 is max disagreement.
    df['obi_consensus'] = 1.0 - (df['obi_spread'] / 2.0)

    # 2. Delta-OBI Alignment: Are aggressive trades moving in the direction of the book?
    # 1.0 = Sync, -1.0 = Absorption/Fighting, 0.0 = Neutral.
    df['delta_obi_alignment'] = np.sign(df['mean_delta']) * np.sign(df['mean_obi'])

    # 3. Momentum Efficiency: How much momentum is generated per unit of volatility?
    # High value = clean trend; Low value = noisy/choppy movement.
    df['momentum_efficiency'] = df['mean_momentum'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Dispersion vs Volatility: Is the cross-exchange spread normal for this regime?
    df['dispersion_ratio'] = df['price_std'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata but keep pm_costs and other features for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome', 'seconds_remaining']
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
    Decide whether to trade using "Consensus-Gated Impulse Scaling (CGIS)".
    The strategy seeks 'Impulse' setups where Delta and OBI are aligned and 
    consensus is high, while penalizing trades near the strike price or 
    during periods of exchange disagreement.
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
    alignment = features_row.get('delta_obi_alignment', 0.0)
    consensus = features_row.get('obi_consensus', 0.0)
    moneyness = features_row.get('price_vs_strike_pct', 0.0)
    efficiency = features_row.get('momentum_efficiency', 0.0)

    # 3. Base Hurdle
    # 5-minute windows are noisy; we need a significant edge.
    hurdle = 0.03
    if best_ev < hurdle:
        return None, 0.0

    # 4. Multiplier Logic (The Core Alpha)
    multiplier = 1.0

    # A. Alignment Check: If Delta and OBI are fighting (Absorption), reduce conviction.
    if alignment <= 0:
        multiplier *= 0.3
    else:
        # B. Consensus Scaling: If exchanges agree, increase conviction.
        # High consensus (near 1.0) can boost multiplier up to 1.5x.
        multiplier *= (1.0 + (0.5 * consensus))

    # C. Moneyness Guard: If price is very close to the strike, the outcome is 
    # extremely sensitive to noise (the "coin flip" zone). Reduce exposure.
    if abs(moneyness) < 0.0005:
        multiplier *= 0.5

    # D. Momentum Efficiency: If momentum is extremely inefficient (choppy), reduce conviction.
    if efficiency < 0.1:
        multiplier *= 0.7

    # 5. Execution & Sizing
    # Time-decay: Reduce exposure as the window closes to avoid late-window volatility.
    time_factor = (seconds_remaining / 300.0)

    # Scale bet_fraction by (Edge / Hurdle) * regime_multiplier * time_decay.
    bet_fraction = (best_ev / hurdle) * multiplier * time_factor
    bet_fraction = min(1.0, max(0.0, bet_fraction))

    # Final safety check: If the multiplier or edge is too low, abstain.
    if bet_fraction < 0.05:
        return None, 0.0

    return side, bet_fraction