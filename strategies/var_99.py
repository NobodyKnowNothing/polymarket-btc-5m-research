# Var 99 | BT: 0.455 | Live PnL: $-0.59 | Trades: 132
# Desc: Implements "Tri-Regime Microstructure Gating (TRMG)," which distinguishes between steady trends, absorption traps, and explosive "liquidity vacuum" breakouts to optimize entry thresholds and bet sizin
# Stats: {"live_returns": [3.303756, 0.9968399405479431, 1.060396671295166, -1.0, 0.0475773811340332, -1.0, -0.6646326184272766, 0.5578039884567261, 1.394787311553955, 0.057208865880966187, 0.8545929193496704, 0.8939649999999999, 0.25678369402885437, 0.33879342675209045, -0.41416096687316895, 0.9305019999999999, 0.07696378231048584, 1.7133684158325195, 0.047612935304641724, -0.7912564873695374, -0.9514586329460144, 1.4562249999999999, -0.49117690324783325, 0.32795625925064087, 0.014042094349861145, -0.26503971219062805, 1.2023675441741943, -0.7530708909034729, -0.4944417476654053, -0.5390887260437012, 4.245841, -0.23394522070884705, 2.429187297821045, 0.04268378019332886, -0.5222631692886353, -0.823637843132019, 0.07716107368469238, -0.5971097946166992, -1.0, -0.3518272936344147, 0.01841600239276886, 0.060449302196502686, -0.8666245341300964, 0.3170856237411499, -1.0, -1.0, -0.16081835329532623, 0.08363932371139526, 0.017779439687728882, -0.10990004241466522, -1.0, -0.871259868144989, -1.0, -1.0, 0.05564397573471069, -0.827890932559967, -0.8690322637557983, -1.0, -1.0, 2.9500198364257812, 0.033746808767318726, -0.9130282402038574, 2.4319279193878174, -0.7913205027580261, 1.146843433380127, -0.27752766013145447, -0.6654782295227051, 3.628821849822998, -1.0, -0.6082869172096252, -1.0, 0.042510032653808594, -1.0, 0.014032319188117981, 2.39541, -0.709804117679596, -0.3559141755104065, -0.9032580256462097, -0.22931039333343506, 0.3936024606227875, -0.960142970085144, 4.550646, -0.34431594610214233, 3.1194439999999997, -1.0, -1.0, -0.7078240513801575, -0.5728414058685303, -0.2574306130409241, -0.9394441246986389, 2.095360040664673, 1.0812182426452637, -0.3754781484603882, -0.6418955326080322, 1.0721967220306396, -1.0, -0.9356043934822083, -1.0, -1.0, -1.0, 1.9791409999999998, 0.03037753701210022, -0.6135731339454651, -1.0, -1.0, -1.0, -1.0, -0.3104268014431, -1.0, -1.0, -0.6944915056228638, -0.37636175751686096, 1.8704757690429688, -1.0, 3.505625, -1.0, -0.9229497909545898, 0.900311291217804, 2.795066, -1.0, 1.7071268558502197, -0.9139451384544373, -0.38478752970695496, 0.013505816459655762, -0.9978777170181274, -0.9640889167785645, -1.0, -1.0, 0.021359622478485107, -0.856361448764801, -1.0, -0.4263421297073364], "live_pnl": -0.5905494093894958, "live_wins": 52, "live_total": 132, "age": 230, "rounds_underwater": 2, "abstain_rounds": 0, "backtest_score": 0.455392651297837, "backtest_acc": 0.5832171083217108, "backtest_n": 3156}

import pandas as pd
import numpy as np
import math

def engineer_features(df):
    """
    Transform raw flat features into model-ready features, focusing on 
    price-flow velocity, order book consensus, and liquidity vacuum detection.
    """
    df = df.copy()

    # 1. Price-Flow Velocity: How fast is the price moving relative to aggressive flow?
    df['price_velocity'] = df.groupby('market_slug')['binance_price'].diff().fillna(0)
    df['delta_acceleration'] = df.groupby('market_slug')['mean_delta'].diff().fillna(0)

    # 2. Order Book Consensus: How much do the exchanges agree on the imbalance?
    # High value = all exchanges seeing similar pressure.
    df['obi_consensus'] = 1.0 / (df['obi_spread'] + 1e-6)

    # 3. Flow Pressure: Directional strength of the combined OBI and Delta.
    df['flow_pressure'] = df['mean_delta'] * df['mean_obi']

    # 4. Liquidity Vacuum Index: Detects sudden delta spikes in low-vol/low-spread environments.
    # This identifies potential "breakout" moments where price is about to move violently.
    df['vacuum_index'] = df['mean_delta'].abs() / (df['mean_volatility'] + df['price_spread'] + 1e-6)

    # Drop metadata and target
    cols_to_drop = ['market_slug', 'timestamp', 'actual_outcome']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Final cleanup
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X

def get_hyperparams():
    """Return highly regularized hyperparameters to prevent overfitting to micro-noise."""
    return {
        'n_estimators': 1200,
        'max_depth': 2,
        'learning_rate': 0.008,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 30.0,
        'reg_lambda': 30.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether to trade using "Tri-Regime Microstructure Gating (TRMG)".

    Logic:
    - Vacuum Regime: High vacuum_index + low volatility. High conviction, low EV threshold.
    - Trend Regime: Price velocity and delta are aligned. Moderate EV threshold.
    - Absorption Regime: Price velocity and delta are opposing. High EV threshold (risk of trap).
    - Neutral Regime: No clear regime. High EV threshold.
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down

    if ev_up > ev_down:
        best_ev = ev_up
        side = 'Up'
        direction = 1
    else:
        best_ev = ev_down
        side = 'Down'
        direction = -1

    # Extract engineered features
    v_idx = features_row.get('vacuum_index', 0.0)
    p_vel = features_row.get('price_velocity', 0.0)
    delta = features_row.get('mean_delta', 0.0)
    vol = features_row.get('mean_volatility', 0.0)
    consensus = features_row.get('obi_consensus', 0.0)

    # 1. Identify Microstructure Regimes
    # Vacuum: Sudden aggressive flow in a quiet market (Breakout signal)
    is_vacuum = v_idx > 2.5 and vol < 0.001 
    
    # Trend: Price and Delta moving in the same direction
    is_trend = (p_vel * delta) > 0
    
    # Absorption: Price moving against the aggressive flow (Reversal/Trap signal)
    is_absorption = (p_vel * delta) < 0

    # 2. Set Regime-Specific Thresholds and Multipliers
    if is_vacuum:
        # Explosive breakout: we want to catch this early
        req_ev = 0.018
        multiplier = 1.2
    elif is_trend:
        # Steady trend: standard momentum play
        req_ev = 0.030
        multiplier = 1.0
    elif is_absorption:
        # Absorption: high risk of being 'run over'; require massive edge
        req_ev = 0.055
        multiplier = 0.4
    else:
        # Noise
        req_ev = 0.040
        multiplier = 0.6

    # 3. Apply Consensus and Time Gating
    # If exchanges are fighting (low consensus), increase the requirement
    if consensus < 0.5:
        req_ev *= 1.3
        multiplier *= 0.7

    # As time runs out, reduce bet size to avoid end-of-window noise
    time_decay = min(1.0, seconds_remaining / 300.0)
    
    if best_ev > req_ev:
        # Scale bet size by edge magnitude and regime multiplier
        # Target 5% EV as the baseline for 1.0 bet_fraction
        bet_fraction = multiplier * (best_ev / 0.05) * time_decay
        return side, min(1.0, max(0.0, bet_fraction))

    return None, 0.0