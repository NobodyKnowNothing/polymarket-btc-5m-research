# Var 65 | BT: 0.956 | Live PnL: $-3.43 | Trades: 61
# Desc: Implements "Convergence-Validated Momentum (CVM)," which targets high-conviction moves by requiring that exchange price dispersion is actively narrowing (convergence) while aggressive flow and order b
# Stats: {"live_returns": [0.3550505605052593, 0.028392319499750462, -0.13564457321166992, -0.18793171628316244, 0.5630618515630563, 0.024295501578481676, 0.00918705160264588, -0.1878250567118327, 0.33939107127789303, 0.01113315023979694, -0.1810201784769694, -0.18797287146250408, 0.0078867866809616, 0.005835777128547659, -0.18776982434590658, -0.18778456242879232, 0.49847098667650486, 0.21377820144218637, -0.18773682053883872, -0.1875754289627075, -0.18752882099151613, -0.18756468391418457, -0.08283677991231284, -0.18789838504791262, -0.1877613836924235, -0.06846749273935954, -0.18777406152089438, -0.08246090602874756, -0.18716498374938967, 0.03896472884167479, 0.22240912312725644, -0.18736757119496664, 0.8548983432833293, -0.18650606377919515, 0.025884614299225783, -0.1867597926457723, -0.1874998343785604, -0.18702310021718344, -0.18764838695526123, -0.18768881384531658, -0.18703306579589846, -0.18766793219248457, 0.011066475155342731, 0.24028216041990913, -0.18751836045583092, 0.27350733792730175, -0.18761238352457685, -0.18690087350209555, -0.1876480801900228, -0.18699155108133952, 0.021566841388867386, -0.18782010904947918, -0.18804527282714845, -0.18751073614756267, -0.18773028151194257, -0.1873063062032064, 0.017227036625212994, -0.18797442944844567, -0.18792365646362308, -0.18746853446960451, -0.07957866954803466], "live_pnl": -3.4316524161834714, "live_wins": 20, "live_total": 61, "age": 88, "rounds_underwater": 39, "abstain_rounds": 0, "backtest_score": 0.9563328730814988, "backtest_acc": 0.5395161290322581, "backtest_n": 2586}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the convergence of exchange prices and the alignment of flow/liquidity.
    """
    df = df.copy()

    # 1. Spread Velocity: Is the disagreement between exchanges shrinking or growing?
    # Negative means exchanges are converging on a single price (high conviction).
    df['spread_velocity'] = df.groupby('market_slug')['price_spread'].diff().fillna(0)

    # 2. Delta-OBI Confluence: Are aggressive trades supported by the book?
    # Positive/Negative product means delta and OBI are moving in the same direction.
    df['delta_obi_confluence'] = df['mean_delta'] * df['mean_obi']

    # 3. Volatility-Normalized Spread: Is the current spread significant relative to noise?
    df['vol_normalized_spread'] = df['price_spread'] / (df['mean_volatility'] + 1e-6)

    # 4. Price Momentum: Recent price velocity
    df['price_momentum'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to noise."""
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
    Decide whether to trade using "Convergence-Validated Momentum (CVM)".
    
    Logic:
    1. High-Conviction Consensus: High EV + Shrinking Spread + Flow/OBI Alignment.
    2. Pure Momentum: High EV + Flow/OBI Alignment (even if spread is wide).
    3. Value Convergence: Moderate EV + Shrinking Spread + Flow/OBI Alignment.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    
    # Determine best direction
    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
    else:
        best_ev = ev_down
        side = 'Down'

    # Extract engineered features
    spread_vel = features_row.get('spread_velocity', 0.0)
    confluence = features_row.get('delta_obi_confluence', 0.0)
    norm_spread = features_row.get('vol_normalized_spread', 1.0)

    # Microstructure signals
    is_converging = spread_vel < 0
    is_aligned = abs(confluence) > 0.02
    is_stable = norm_spread < 0.25

    # Time-gate: Avoid the extreme noise of the final 45 seconds
    if seconds_remaining < 45:
        return None, 0.0

    # Decision Matrix
    bet_fraction = 0.0

    # Case 1: High-Conviction Consensus (The "Gold Standard")
    if best_ev > 0.04 and is_converging and is_aligned and is_stable:
        bet_fraction = 1.0
        
    # Case 2: Pure Momentum (Aggressive flow is strong, even if spread is messy)
    elif best_ev > 0.04 and is_aligned:
        bet_fraction = 0.4
        
    # Case 3: Value Convergence (Moderate edge, but the market is settling)
    elif best_ev > 0.025 and is_converging and is_aligned:
        bet_fraction = 0.6

    if bet_fraction > 0:
        # Scale down slightly as time runs out to reduce variance
        time_factor = min(1.0, seconds_remaining / 300.0)
        return side, bet_fraction * time_factor

    return None, 0.0