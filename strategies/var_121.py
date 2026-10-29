# Var 121 | BT: -1.224 | Live PnL: $-1.50 | Trades: 65
# Desc: Implements "Coiled Spring & Exhaustion Gating (CSEG)," which uses the log-transformed ratio of Order Book Imbalance velocity to Price velocity to amplify "coiled" breakout signals (high OBI movement w
# Stats: {"live_returns": [-0.34469911456108093, -1.0, 3.1194439999999997, -0.5310187339782715, -0.509365975856781, -1.0, -1.0, 2.879950761795044, 0.5770387649536133, 3.9731249999999996, -0.8751433491706848, 0.09457680583000183, 0.749462366104126, 1.559455394744873, 0.7637473344802856, -0.936409056186676, -0.9551084637641907, -0.2099166363477707, 4.893558, -0.32084211707115173, -0.3082242012023926, -0.3104836046695709, -0.8451313376426697, -0.3388839364051819, -0.4350900948047638, -0.30447840690612793, 2.1217970848083496, -0.7945705652236938, 2.593877077102661, -0.8792296051979065, 1.001373529434204, -0.5498223900794983, -0.38039305806159973, -0.3852607011795044, 0.0943303108215332, -0.8541858792304993, -0.7758163809776306, -0.28345027565956116, -0.44823792576789856, -0.9713060259819031, -0.8784071207046509, 0.37575334310531616, -0.5261235237121582, -1.0, -0.21635253727436066, 0.04931366443634033, 1.184525489807129, 2.3817052841186523, -1.0, -0.34516432881355286, -1.0, -1.0, -0.33918651938438416, -1.0, -1.0, 1.7294070000000001, -0.29118579626083374, -1.0, -1.0, -1.0, 0.02950039505958557, -0.6932101249694824, -0.8440452814102173, -1.0, -0.9958978295326233], "live_pnl": -1.504699945449829, "live_wins": 19, "live_total": 65, "age": 128, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -1.223829622648138, "backtest_acc": 0.546524064171123, "backtest_n": 3255}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the relationship between order book momentum and price velocity.
    """
    # Drop metadata but keep pm_costs and seconds_remaining for model context
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

    # 1. Calculate Velocities (grouped by market to prevent leakage)
    # OBI Velocity: How fast the order book imbalance is shifting
    X['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    
    # Price Velocity: How fast the price is actually moving
    X['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)

    # 2. Velocity Ratio (The "Coiled Spring" metric)
    # We use log1p to compress the range and handle high-volatility spikes.
    # High ratio = OBI moving fast, price stalling (Coiled Spring).
    # Low ratio = Price moving fast, OBI stalling (Exhaustion/Slippage).
    X['velocity_ratio'] = np.log1p(
        X['obi_velocity'].abs() / (X['price_velocity'].abs() + 1e-6)
    )

    # 3. Delta-Price Efficiency
    # Measures how much aggressive delta is required to move the price 1 unit.
    X['delta_efficiency'] = X['sum_delta'].abs() / (X['price_velocity'].abs() + 1e-6)

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters to prioritize signal over noise."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.005,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'reg_alpha': 35.0,
        'reg_lambda': 55.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using Coiled Spring & Exhaustion Gating (CSEG).
    
    This strategy identifies 'coiled' markets where order book pressure is 
    building up without immediate price movement, suggesting a high-conviction 
    breakout is imminent. It avoids 'exhaustion' moves where price is 
    running on momentum without order book support.
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

    # 2. Extract Microstructure Metrics
    # log_ratio > 2.0 implies a massive OBI movement relative to price (Coiled Spring)
    # log_ratio < 0.5 implies price is moving significantly faster than OBI (Exhaustion)
    log_ratio = features_row.get('velocity_ratio', 0.0)
    
    # 3. Apply CSEG Multiplier
    # We scale the 'conviction' of the model based on the microstructure state.
    if log_ratio > 2.0:
        # Coiled Spring: High conviction breakout signal
        conviction_multiplier = 1.5
    elif log_ratio < 0.5:
        # Exhaustion: High risk of a reversal or liquidity gap
        conviction_multiplier = 0.4
    else:
        # Normal regime
        conviction_multiplier = 1.0

    effective_ev = best_ev * conviction_multiplier

    # 4. Execution Logic
    # Base hurdle is 2.5% to account for market noise and spread.
    base_hurdle = 0.025

    if effective_ev > base_hurdle:
        # Sizing: Scale bet size by the magnitude of effective edge.
        # We use a conservative 0.5 multiplier to ensure we don't over-leverage.
        bet_fraction = (effective_ev / base_hurdle) * 0.5

        # Time-decay: Reduce exposure as the 5-minute window closes to avoid last-second volatility.
        time_factor = min(1.0, seconds_remaining / 300.0)
        bet_fraction *= time_factor

        # Final safety bounds
        bet_fraction = min(1.0, max(0.0, bet_fraction))

        # Only execute if the bet is meaningful
        if bet_fraction > 0.02:
            return side, bet_fraction

    return None, 0.0