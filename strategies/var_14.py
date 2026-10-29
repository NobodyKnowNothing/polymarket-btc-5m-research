# Var 14 | BT: -0.138 | Live PnL: $-1.60 | Trades: 61
# Desc: Implements "Asymmetric Flow-Price Divergence (AFPD) Gating," which applies a higher hurdle to trades occurring during order-flow absorption (where delta and price move in opposite directions) and uses
# Stats: {"live_returns": [1.4067401885986328, 0.7038716673851013, -0.36861997842788696, -0.491743266582489, 1.4201000928878784, -0.21089217066764832, -0.17451338469982147, -0.33848485350608826, 1.2894476652145386, -0.4291726052761078, -0.4109848439693451, 0.686424732208252, 0.7719393968582153, 0.36677688360214233, -0.3315814435482025, -0.46562522649765015, -0.37596696615219116, -0.2095533013343811, -0.17153480648994446, -0.23067080974578857, -0.24283941090106964, -0.5213736891746521, 1.108433485031128, -0.33093634247779846, -0.6766594052314758, 2.950496, -1.0, 1.7645650000000002, -1.0, -0.6468305587768555, -0.8170491456985474, -0.4543757438659668, -0.41750991344451904, -0.686429500579834, -1.0, -0.7949845194816589, -0.7895796298980713, -1.0, 0.047107040882110596, -1.0, -1.0, -1.0, 1.4562249999999999, -1.0, -1.0, -1.0, 0.07465547323226929, -1.0, 1.890848, 2.651597, -1.0, -0.7895796298980713, -0.6217831373214722, -0.5070969462394714, -0.8439757227897644, -1.0, 1.564096212387085, 7.409003257751465, -1.0, -1.0, -0.8077250123023987], "live_pnl": -1.5957441329956055, "live_wins": 17, "live_total": 61, "age": 125, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -0.13798682068386753, "backtest_acc": 0.5646229438154241, "backtest_n": 3101}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    detecting order-flow absorption and price-flow alignment.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Flow-Price Alignment: Measures if aggressive delta and price move together.
    # 1.0 = Trend (confluence), -1.0 = Absorption (divergence), 0.0 = Neutral.
    X['flow_price_alignment'] = np.sign(X['sum_delta']) * np.sign(X['price_vs_strike_pct'])

    # 2. Absorption Index: Quantifies how much delta is required to move the price.
    # High values indicate a "wall" of liquidity absorbing aggressive trades.
    X['absorption_index'] = X['sum_delta'].abs() / (X['price_vs_strike_pct'].abs() + 1e-5)

    # 3. OBI-Momentum Confluence: Checks if order book imbalance aligns with recent price action.
    X['obi_momentum_confluence'] = X['mean_obi'] * X['mean_momentum']

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 25.0,
        'reg_lambda': 45.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Asymmetric Flow-Price Divergence (AFPD) Gating.
    The hurdle increases significantly when the market is in an 'absorption' state 
    (high delta, low price movement) to avoid being caught in liquidity traps.
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

    # 2. Extract engineered quality metrics
    alignment = features_row.get('flow_price_alignment', 1.0)
    absorption = features_row.get('absorption_index', 0.0)
    volatility = features_row.get('mean_volatility', 0.01)

    # 3. Calculate Dynamic Asymmetric Hurdle
    # Base hurdle is 2%.
    base_hurdle = 0.02
    
    # Penalty for Divergence: If delta and price are fighting (alignment < 0), 
    # the risk of a fake-out is much higher.
    divergence_penalty = 0.025 if alignment < 0 else 0.0
    
    # Penalty for Extreme Absorption: If the absorption index is very high, 
    # add an extra layer of caution.
    absorption_penalty = 0.015 if absorption > 5.0 else 0.0
    
    # Volatility penalty: Scale hurdle with the current volatility regime.
    vol_penalty = 0.03 * volatility

    dynamic_hurdle = base_hurdle + divergence_penalty + absorption_penalty + vol_penalty

    # 4. Execution Logic
    if best_ev > dynamic_hurdle:
        # Sizing: Use a Volatility-Capped Kelly-style fraction.
        # As volatility increases, we aggressively reduce the bet fraction to protect Sharpe.
        bet_fraction = (best_ev / dynamic_hurdle) * (1.0 / (1.0 + volatility))
        
        # Apply a small time-decay to reduce exposure in the final seconds of the window
        time_factor = max(0.5, min(1.0, seconds_remaining / 300.0))
        bet_fraction *= time_factor

        # Final safety checks
        bet_fraction = min(1.0, max(0.0, bet_fraction))
        
        if bet_fraction > 0.05:
            return side, bet_fraction

    return None, 0.0