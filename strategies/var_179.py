# Var 179 | BT: -0.191 | Live PnL: $-0.11 | Trades: 142
# Desc: Implements "Lead-Lag Crowding Filter (LLCF)," which uses the price divergence between Binance and other exchanges as a leading signal, while using a combined Funding-OBI index to avoid entering "crowd
# Stats: {"live_returns": [-1.0, -1.0, -1.0, -1.0, 9.427877, 1.5191249999999998, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.7809233069419861, -1.0, -1.0, -0.7830168604850769, -1.0, 1.6554160000000002, -1.0, -1.0, -1.0, -1.0, 1.3964720000000002, 0.15118699999999996, 0.025412559509277344, 12.389175, -1.0, -1.0, 0.15669173002243042, -1.0, -1.0, -1.0, -1.0, 1.5854279999999998, -1.0, 1.1375739999999999, -1.0, 0.05923210000000001, 0.5527185797691345, 2.39541, -1.0, 1.5854279999999998, -1.0, -0.7623190879821777, 2.950496, 1.3396370000000002, 1.7294070000000001, -0.05, -0.05, -1.0, -0.05, -1.0, 2.173394, -0.05, -0.05, -0.05, -0.05, -1.0, -1.0, 0.1559722, -0.05, 0.09194362163543701, 1.890848, -1.0, -0.05, 0.05462775, -1.0, -1.0, 0.09454240000000001, 0.13257985, -0.05, -0.05, 0.03692430257797241, -0.4727378487586975, -1.0, -1.0, -0.05, -1.0, -1.0, 0.04523050785064697, 2.173394, -1.0, -0.05, -0.05, -1.0, 0.0903876, -0.05, -1.0, -0.05, 2.518758, 2.651597, -1.0, -0.05, -0.05, 1.4562249999999999, -0.05, -1.0, -1.0, -0.7966662049293518, 0.17528125, -1.0, -0.05, -0.05, -1.0, -1.0, -1.0, 3.727685, -1.0, -1.0, -0.8972508907318115, -0.05, -1.0, -0.05, -0.05, -0.05, -0.05, 0.11402865, -0.05, 0.0051583999999999935, -0.05, -0.05, 0.005738350000000003, 0.07595624999999999, -0.05, -0.05, 0.05462775, 0.0069382999999999945, -0.05, 0.05687869999999999, 0.05923210000000001, 0.09454240000000001, 0.0616951, -0.05, -0.05, -0.05, 0.0616951, 0.09454240000000001, 0.06698185000000001, -0.05, -0.05, -0.05, -1.0], "live_pnl": -0.10882356166839613, "live_wins": 46, "live_total": 142, "age": 372, "rounds_underwater": 1, "abstain_rounds": 0, "backtest_score": -0.1905698776419555, "backtest_acc": 0.5400182315405652, "backtest_n": 2291}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    Binance price leadership and liquidity-adjusted momentum.
    """
    df = df.copy()

    # 1. Binance Lead-Lag: The percentage difference between Binance price 
    # and the average of Coinbase/Kraken. Binance is the primary price driver.
    other_avg_price = (df['coinbase_price'] + df['kraken_price']) / 2
    df['binance_lead_pct'] = (df['binance_price'] - other_avg_price) / df['binance_price']

    # 2. Crowding Index: A proxy for market positioning.
    # High positive = heavily long (risk of squeeze); High negative = heavily short.
    # We scale funding to be comparable to OBI magnitude.
    df['crowding_index'] = df['mean_obi'] + (df['mean_funding'] * 1000)

    # 3. Volatility-Adjusted Lead: Normalizes the Binance lead by current volatility.
    df['vol_adj_lead_pct'] = df['binance_lead_pct'] / (df['mean_volatility'] + 1e-6)

    # 4. Delta Efficiency: How much price moves per unit of aggressive flow.
    # Low efficiency = high absorption (reversal signal).
    df['delta_efficiency'] = df['mean_delta'] / (df['price_std'] + 1e-6)

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
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.007,
        'subsample': 0.6,
        'colsample_bytree': 0.5,
        'reg_alpha': 40.0,
        'reg_lambda': 40.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Lead-Lag Crowding Filter (LLCF)".

    Logic:
    1. Identify Lead Direction: Is Binance leading the price up or down relative to others?
    2. Check Crowding: Is the trade direction already "crowded" (high OBI + high funding)?
       If we want to go 'Up' but crowding is > 0.6, we abstain to avoid liquidations.
    3. Confirm with Model: Only trade if the ML model's predicted EV is significant.
    4. Dynamic Sizing: Scale bet size by the strength of the lead and the lack of crowding.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    # Extract engineered features
    lead_pct = features_row.get('binance_lead_pct', 0.0)
    vol_adj_lead = features_row.get('vol_adj_lead_pct', 0.0)
    crowding = features_row.get('crowding_index', 0.0)

    side = None
    best_ev = 0.0
    
    # Base hurdle for EV
    hurdle = 0.035

    # 1. Directional Logic based on Binance Leadership
    if lead_pct > 0:
        # Binance is leading UP. We look for 'Up' trades.
        # BUT, if crowding is too high, the move is exhausted/risky.
        if ev_up > hurdle and crowding < 0.6:
            side = 'Up'
            best_ev = ev_up
            
    elif lead_pct < 0:
        # Binance is leading DOWN. We look for 'Down' trades.
        # BUT, if crowding is too low (very negative), the move is exhausted/risky.
        if ev_down > hurdle and crowding > -0.6:
            side = 'Down'
            best_ev = ev_down

    # 2. Execution and Sizing
    if side:
        # Scale confidence by the volatility-adjusted lead strength
        # We cap confidence at 1.0
        confidence = min(1.0, abs(vol_adj_lead) * 5.0)
        
        # Scale by how "un-crowded" the market is (0.0 to 1.0)
        crowding_buffer = max(0.1, 1.0 - abs(crowding))
        
        # Scale by the magnitude of the edge (target 5% EV for full bet)
        edge_scale = best_ev / 0.05
        
        # Time decay: reduce size as the window closes to avoid last-second whipsaws
        time_factor = 1.0 if seconds_remaining > 30 else (seconds_remaining / 30.0)

        bet_fraction = confidence * crowding_buffer * edge_scale * time_factor
        
        # Ensure bet is within reasonable bounds
        return side, min(1.0, max(0.05, bet_fraction))

    return None, 0.0