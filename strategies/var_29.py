# Var 29 | BT: -1.134 | Live PnL: $-4.99 | Trades: 61
# Desc: Implements "Tri-Exchange Flow-Imbalance Velocity (T-FIV)," which requires directional agreement across all three exchanges' order books and validates that aggressive flow velocity is driving the imbal
# Stats: {"live_returns": [-0.16782014071941376, 0.1903177946805954, -0.49424514174461365, -0.10073941200971603, -0.12973260879516602, -0.9197438955307007, -0.16018502414226532, -0.34805408120155334, -0.4105631113052368, -0.20599496364593506, -0.2751070559024811, -0.16722513735294342, -0.5059248208999634, -1.0, 0.45683684945106506, -0.20534949004650116, 0.387467622756958, -0.13619031012058258, -1.0, -0.5047071576118469, 0.716522216796875, 0.8701651096343994, -0.11323006451129913, -0.14612703025341034, 0.31825411319732666, -0.1347140669822693, 1.0038130283355713, 0.8833919763565063, -0.13165482878684998, -1.0, 0.31278204917907715, -0.33595800399780273, -0.3223486840724945, -0.11252976208925247, -0.13889186084270477, 0.3854644298553467, -0.3052872121334076, -0.5962158441543579, -0.5033277273178101, -0.5244524478912354, -1.0, 0.5132893323898315, -0.4012579917907715, 0.7329120635986328, -0.1352967470884323, 0.5404345989227295, 0.5475642681121826, 0.38717129826545715, -0.2528758943080902, -0.2886570692062378, 0.05375131964683533, -0.13453905284404755, -0.15683235228061676, -0.14130353927612305, -0.6517608761787415, -0.17548500001430511, 0.4367823302745819, -0.21646586060523987, 0.7824610471725464, 0.2763577699661255, -0.13351942598819733], "live_pnl": -4.988575458526611, "live_wins": 19, "live_total": 61, "age": 190, "rounds_underwater": 59, "abstain_rounds": 0, "backtest_score": -1.133576896502474, "backtest_acc": 0.5772192946899067, "backtest_n": 610}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    velocity of flow-book alignment and multi-exchange OBI agreement.
    """
    df = df.copy()
    
    # 1. OBI Agreement: How many exchanges agree on the direction of imbalance?
    # (1.0 if all three signs match, 0.66 if two match)
    s1 = np.sign(df['binance_obi'])
    s2 = np.sign(df['coinbase_obi'])
    s3 = np.sign(df['kraken_obi'])
    df['obi_agreement'] = np.where((s1 == s2) & (s2 == s3), 1.0, 0.66)

    # 2. Velocity Features: Capturing the rate of change in flow and imbalance
    # We must group by market_slug to prevent cross-market leakage
    df['delta_velocity'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)
    df['obi_velocity'] = df.groupby('market_slug')['mean_obi'].diff().fillna(0)
    
    # 3. Price Consensus: Inverse of cross-exchange dispersion
    df['price_consensus'] = 1.0 / (1.0 + df['price_std'])

    # Drop metadata but keep pm_costs and seconds_remaining for the model
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.fillna(0).replace([float('inf'), float('-inf')], 0)

    return X

def get_hyperparams():
    """Return highly regularized XGBoost hyperparameters."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 20.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "T-FIV" logic.
    Requires high OBI agreement, price consensus, and non-absorptive flow velocity.
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
    agreement = features_row.get('obi_agreement', 0.66)
    consensus = features_row.get('price_consensus', 0.5)
    d_vel = features_row.get('delta_velocity', 0.0)
    o_vel = features_row.get('obi_velocity', 0.0)
    vol = features_row.get('mean_volatility', 0.01)
    std = features_row.get('price_std', 0.0)

    # 3. Dynamic Hurdle Calculation
    # Base hurdle increases with volatility and price dispersion
    hurdle = 0.025 + (0.03 * vol) + (0.02 * std)

    # Penalty A: Absorption Detection
    # If delta velocity and OBI velocity are opposing, flow is being absorbed.
    # This is a high-risk scenario for momentum-based models.
    if d_vel * o_vel < 0:
        hurdle += 0.04

    # Penalty B: Disagreement
    # If OBI agreement is not perfect (1.0), demand a higher edge.
    if agreement < 1.0:
        hurdle += 0.02

    # 4. Execution Logic
    if best_ev > hurdle:
        # Time-decay: Scale down exposure as the window closes to avoid noise.
        time_factor = math.sqrt(max(0.0, min(1.0, seconds_remaining / 300.0)))

        # Confidence Scaling: 
        # Scale by (Edge/Hurdle) * Agreement * Consensus * Time
        bet_fraction = (best_ev / hurdle) * agreement * consensus * time_factor
        bet_fraction = min(1.0, bet_fraction)

        # Minimum threshold for meaningful trades
        if bet_fraction < 0.1:
            return None, 0.0

        return side, bet_fraction

    return None, 0.0