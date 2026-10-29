# Var 68 | BT: -1.117 | Live PnL: $-1.00 | Trades: 95
# Desc: Implements "Liquidity-Sweep Consensus (LSC)," which targets high-conviction moves where aggressive flow pierces thin order books (high sweep index) and is validated by cross-exchange price agreement (
# Stats: {"live_returns": [0.6569103002548218, 0.5435590147972107, -0.47653600573539734, -0.5684954524040222, -0.2588745355606079, -0.04175955802202225, -0.1675124615430832, -0.8375130891799927, 0.4504324793815613, -0.5095066428184509, -0.7055134773254395, -0.48230576515197754, -0.3797304630279541, -0.11429203301668167, 0.008418068289756775, -0.20200103521347046, 2.0148050785064697, -0.3496250808238983, 0.07598526775836945, 0.6164750456809998, -0.18147701025009155, -0.04326909780502319, 0.7438691854476929, 1.5521847009658813, -0.06447577476501465, 0.005022972822189331, -0.08267635852098465, 0.9154845476150513, -0.15324610471725464, -0.28326669335365295, 1.0190341472625732, -0.08082539588212967, 0.06692920625209808, 0.09094546735286713, -0.1524585336446762, -0.07200474292039871, -0.4686834216117859, 0.43901291489601135, -0.31069743633270264, -0.09426544606685638, -0.20474980771541595, -0.26574331521987915, -0.12403536587953568, -0.033315882086753845, -0.14826232194900513, 0.7812610864639282, -0.2412819266319275, -0.34545010328292847, -0.2452051043510437, -0.4945639669895172, 0.22289876639842987, -0.11833781003952026, 0.3401623070240021, 0.22479966282844543, 0.1469409167766571, -0.5015833377838135, 0.014159277081489563, -0.5963705778121948, -1.0, -0.050087738782167435, -0.396257221698761, 4.3740057945251465, -0.11025063693523407, 2.604825735092163, -0.13824422657489777, -0.30417758226394653, -0.9154708981513977, -0.5005221366882324, -0.11144810914993286, -0.1469639539718628, -0.445310115814209, -0.18036317825317383, -0.7830803990364075, 1.1986082792282104, 0.003581039607524872, 0.09037423133850098, -0.1718769073486328, -0.660631000995636, 1.1374167203903198, -0.19744181632995605, -0.6189954876899719, -0.07713203877210617, -0.12053743004798889, -0.11301318556070328, -0.45160219073295593, -0.25768765807151794, 0.7417337894439697, -0.09108700603246689, -0.8790478706359863, -1.0, -0.08421128243207932, 0.016605615615844727, -0.8236458897590637, -0.9227668046951294, -0.14271578192710876], "live_pnl": -0.9980380535125732, "live_wins": 29, "live_total": 95, "age": 130, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": -1.1166412089094653, "backtest_acc": 0.5799892990904227, "backtest_n": 3026}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    the 'Sweep Index' (normalized delta relative to book thickness) 
    and 'Price Consensus' (cross-exchange agreement).
    """
    df = df.copy()

    # 1. Sweep Index: Measures how much aggressive flow is 'piercing' the book.
    # We normalize delta by volatility to get a 'unit' of movement, 
    # then divide by the absolute OBI (liquidity buffer).
    # High value = aggressive flow hitting a thin book (potential breakout).
    # Low value = aggressive flow being absorbed by a thick book.
    norm_delta = df['mean_delta'] / (df['mean_volatility'] + 1e-6)
    df['sweep_index'] = np.log1p(norm_delta.abs()) / (df['mean_obi'].abs() + 0.1)

    # 2. Price Consensus: Measures how much the exchanges agree on the price.
    # Low spread = high consensus. High spread = high uncertainty/fragmentation.
    df['price_consensus'] = 1.0 / (1.0 + df['price_spread'])

    # 3. Volatility-Normalized Delta: Standardize flow by current regime.
    df['vol_norm_delta'] = df['mean_delta'] / (df['mean_volatility'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to microstructure noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 35.0,
        'reg_lambda': 35.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Liquidity-Sweep Consensus (LSC)".

    Logic:
    - Sweep Regime: High sweep_index (flow piercing thin book). 
      Allows for lower EV threshold (0.025) and higher conviction.
    - Absorption Regime: Low sweep_index (flow hitting thick book). 
      Requires much higher EV (0.045) to overcome potential reversal.
    - Consensus Gating: Both regimes scale bet size by cross-exchange price agreement.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        side, best_ev = 'Up', ev_up
    else:
        side, best_ev = 'Down', ev_down

    # Extract engineered features
    sweep_idx = features_row.get('sweep_index', 0.0)
    consensus = features_row.get('price_consensus', 0.0)

    # 1. Identify Regime
    # A sweep index > 5.0 indicates significant flow relative to book depth
    is_sweep = sweep_idx > 5.0

    # 2. Set Regime-Specific Thresholds and Base Scaling
    if is_sweep:
        # High conviction in 'sweeps'; lower threshold to capture the momentum
        req_ev = 0.025
        base_mult = 1.0
    else:
        # High risk of 'absorption' (price stalling); require higher edge
        req_ev = 0.045
        base_mult = 0.5

    # 3. Calculate Bet Fraction
    # Scale by consensus (agreement) and sweep strength (if in sweep regime)
    if is_sweep:
        # scale conviction by how much the sweep exceeds the threshold
        conviction = min(1.0, sweep_idx / 20.0)
        bet_fraction = base_mult * consensus * (0.5 + 0.5 * conviction)
    else:
        bet_fraction = base_mult * consensus

    # 4. Time-Gating and Final Decision
    # As the window closes, reduce bet size to avoid high-volatility noise
    if seconds_remaining < 60:
        bet_fraction *= 0.5
        req_ev *= 1.3

    if best_ev > req_ev:
        # Scale the final bet by the magnitude of the edge relative to the target
        return side, min(1.0, bet_fraction * (best_ev / 0.04))

    return None, 0.0