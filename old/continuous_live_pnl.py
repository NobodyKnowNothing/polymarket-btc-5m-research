import time
import pickle
import pandas as pd
import numpy as np
import warnings
from collections import deque
from main import PolymarketBTC5m
from data_sources import get_btc_features
from train_model import parse_features
from paper_trade import position_size_detailed

warnings.filterwarnings("ignore")

def resolve_market_robustly(market, max_attempts=12) -> str | None:
    """
    Wait for the oracle to settle and return the correct outcome ('Up' or 'Down').
    Uses the CLOB midpoint prices to detect official contract settlement (0.995/0.005).
    """
    from main import PolymarketBTC5m
    slug = market.slug
    print(f"[{time.strftime('%H:%M:%S')}] Awaiting robust resolution for {slug}...")
    
    # Wait initially for 12 seconds
    time.sleep(12)
    
    for attempt in range(max_attempts):
        try:
            # Create a clean market instance for the expired ts
            temp_market = PolymarketBTC5m(initial_ts=market.current_ts)
            state = temp_market.get_prices()
            
            prices = state.get("prices") or state.get("outcome", {})
            up_mid = prices.get("Up", {}).get("midpoint", 0.0)
            down_mid = prices.get("Down", {}).get("midpoint", 0.0)
            
            print(f"    Resolution attempt {attempt+1}: Up mid = {up_mid:.3f}, Down mid = {down_mid:.3f}")
            
            if up_mid > 0.9 and down_mid < 0.1:
                return "Up"
            elif down_mid > 0.9 and up_mid < 0.1:
                return "Down"
                
        except Exception as e:
            print(f"    Error in robust resolution attempt {attempt+1}: {e}")
            
        time.sleep(5)
        
    # Fallback if oracle is extremely slow or something is wrong
    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Robust resolution timed out for {slug}. Falling back to midpoint comparison.")
    try:
        temp_market = PolymarketBTC5m(initial_ts=market.current_ts)
        state = temp_market.get_prices()
        prices = state.get("prices") or state.get("outcome", {})
        up_mid = prices.get("Up", {}).get("midpoint", 0.0)
        down_mid = prices.get("Down", {}).get("midpoint", 0.0)
        if up_mid > down_mid:
            return "Up"
        elif down_mid > up_mid:
            return "Down"
    except Exception:
        pass
    return None


def continuous_live_pnl(margin=0.03):
    print("Loading XGBoost models...")
    try:
        with open("final_model.pkl", "rb") as f:
            saved_data_old = pickle.load(f)
        clf_old = saved_data_old['model']
        calibrator_old = saved_data_old['calibrator']
        expected_features_old = saved_data_old['features']
    except FileNotFoundError:
        print("Could not find final_model.pkl. Make sure you trained the base model.")
        return

    try:
        with open("trend_model.pkl", "rb") as f:
            saved_data_trend = pickle.load(f)
        clf_trend = saved_data_trend['model']
        calibrator_trend = saved_data_trend['calibrator']
        expected_features_trend = saved_data_trend['features']
    except FileNotFoundError:
        print("Could not find trend_model.pkl. Make sure you trained the trend model.")
        return
    
    market = PolymarketBTC5m()
    
    stats_old = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'peak_pnl': 0.0, 'max_drawdown': 0.0}
    stats_trend = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'peak_pnl': 0.0, 'max_drawdown': 0.0}
    BET_SIZE = 10.0  # $10 bet sizing to simulate slippage
    MIN_BOOK_DEPTH_USD = 10.0
    
    print(f"\n=============================================")
    print(f"Starting Continuous Live PnL Tracker")
    print(f"EV Margin required to trade: {margin}")
    print(f"Bet Size per trade: ${BET_SIZE:.2f}")
    print(f"=============================================\n")

    history = deque(maxlen=7)
    traded_this_market = False
    trade_old = {'invested': 0.0, 'prediction': None, 'shares': 0.0, 'thin_book': False}
    trade_trend = {'invested': 0.0, 'prediction': None, 'shares': 0.0, 'thin_book': False}

    while True:
        # 1) Always poll features every 10 seconds to build the history trend
        raw_features = get_btc_features()
        parsed = parse_features(raw_features)
        if parsed:
            history.append(parsed)
            
        secs_remaining = market.market_end_time - time.time()
        
        # 2) Wait if market is expired or we haven't built enough history
        if market.is_expired:
            print(f"[{time.strftime('%H:%M:%S')}] Market expired. Resolving...")
            
            # Resolve robustly by waiting for the oracle to settle
            actual = resolve_market_robustly(market)
            print(f"[{time.strftime('%H:%M:%S')}] {market.slug} Resolved: {actual}")
            
            # Calculate PnL if we traded
            def calc_pnl(name, trade_info, stats):
                if trade_info['invested'] > 0 and trade_info['prediction']:
                    stats['trades'] += 1
                    correct = (trade_info['prediction'] == actual)
                    revenue = trade_info['shares'] * 1.00 if correct else 0.0
                    profit = revenue - trade_info['invested']
                    
                    stats['pnl'] += profit
                    if correct: stats['wins'] += 1
                    else: stats['losses'] += 1
                    
                    if stats['pnl'] > stats['peak_pnl']:
                        stats['peak_pnl'] = stats['pnl']
                        
                    drawdown = stats['peak_pnl'] - stats['pnl']
                    if drawdown > stats['max_drawdown']:
                        stats['max_drawdown'] = drawdown
                    
                    win_rate = (stats['wins'] / stats['trades']) if stats['trades'] > 0 else 0.0
                    print(f"  📈 [{name}] STATS: PnL=${stats['pnl']:.2f} | Max DD=${stats['max_drawdown']:.2f} | Wins={stats['wins']}/{stats['trades']} ({win_rate:.1%})")

            calc_pnl("OLD", trade_old, stats_old)
            calc_pnl("TREND", trade_trend, stats_trend)
            
            print("-" * 60)
            
            # Reset for next market
            market.get_prices() # force advance
            market = PolymarketBTC5m()
            traded_this_market = False
            trade_old = {'invested': 0.0, 'prediction': None, 'shares': 0.0, 'thin_book': False}
            trade_trend = {'invested': 0.0, 'prediction': None, 'shares': 0.0, 'thin_book': False}
            continue
            
        # 3) Trade condition
        if secs_remaining < 270 and not traded_this_market and len(history) >= 7:
            slug = market.slug
            print(f"[{time.strftime('%H:%M:%S')}] Analyzing {slug} with 60s trend context...")
            
            # Compute live trend features
            current_features = history[-1].copy()
            trend_cols = ['mean_obi', 'mean_delta', 'mean_momentum', 'binance_price', 'mean_funding', 'price_std']
            
            for col in trend_cols:
                if col in current_features:
                    vals = [h.get(col, 0) for h in history]
                    current_features[f'{col}_diff_3'] = vals[-1] - vals[-4]
                    current_features[f'{col}_diff_6'] = vals[-1] - vals[0]
                    current_features[f'{col}_roll_mean_6'] = np.mean(vals[-6:])
                    
                    old_val = vals[0]
                    current_features[f'{col}_roc_6'] = (vals[-1] - old_val) / (abs(old_val) + 1e-5)
            
            # 1. Prepare OLD model features
            X_test_old = pd.DataFrame([history[-1].copy()])
            X_test_old['seconds_remaining'] = max(0, secs_remaining)
            for col in expected_features_old:
                if col not in X_test_old.columns:
                    X_test_old[col] = 0.0
            X_test_old = X_test_old[expected_features_old]
            
            # 2. Prepare TREND model features
            X_test_trend = pd.DataFrame([current_features])
            X_test_trend['seconds_remaining'] = max(0, secs_remaining)
            for col in expected_features_trend:
                if col not in X_test_trend.columns:
                    X_test_trend[col] = 0.0
            X_test_trend = X_test_trend[expected_features_trend]
            
            # Inference Old
            y_prob_raw_old = clf_old.predict_proba(X_test_old)[:, 1]
            prob_up_old = calibrator_old.predict(y_prob_raw_old)[0]
            prob_down_old = 1.0 - prob_up_old

            # Inference Trend
            y_prob_raw_trend = clf_trend.predict_proba(X_test_trend)[:, 1]
            prob_up_trend = calibrator_trend.predict(y_prob_raw_trend)[0]
            prob_down_trend = 1.0 - prob_up_trend
            
            # --- REALISTIC PRE-EXECUTION SIMULATION ---
            # Fetch realistic fill profiles first to calculate True EV
            fill_up = position_size_detailed(BET_SIZE, outcome="Up", market=market)
            fill_down = position_size_detailed(BET_SIZE, outcome="Down", market=market)
            
            true_cost_up = fill_up["avg_fill_price"] if fill_up["shares"] > 0 else 1.0
            true_cost_down = fill_down["avg_fill_price"] if fill_down["shares"] > 0 else 1.0
            
            # Evaluate Old
            ev_up_old = prob_up_old - true_cost_up
            ev_down_old = prob_down_old - true_cost_down
            pred_old = None
            if ev_up_old > ev_down_old and ev_up_old > margin: pred_old = "Up"
            elif ev_down_old > ev_up_old and ev_down_old > margin: pred_old = "Down"

            # Evaluate Trend
            ev_up_trend = prob_up_trend - true_cost_up
            ev_down_trend = prob_down_trend - true_cost_down
            pred_trend = None
            if ev_up_trend > ev_down_trend and ev_up_trend > margin: pred_trend = "Up"
            elif ev_down_trend > ev_up_trend and ev_down_trend > margin: pred_trend = "Down"
                
            print(f"  [OLD]   Prob: Up={prob_up_old:.3f}, Down={prob_down_old:.3f} | True EV: Up={ev_up_old:+.3f}, Down={ev_down_old:+.3f} | Pred: {pred_old}")
            print(f"  [TREND] Prob: Up={prob_up_trend:.3f}, Down={prob_down_trend:.3f} | True EV: Up={ev_up_trend:+.3f}, Down={ev_down_trend:+.3f} | Pred: {pred_trend}")
            print(f"  True Cost (w/ fees & slippage): Up={true_cost_up:.3f}, Down={true_cost_down:.3f}")
            
            # --- EXECUTION COMMIT ---
            def commit_execution(pred, trade_info, name):
                if pred:
                    fill = fill_up if pred == "Up" else fill_down
                    trade_info['shares'] = fill["shares"]
                    trade_info['invested'] = BET_SIZE - fill["unfilled"]
                    trade_info['thin_book'] = fill["book_depth_usd"] < MIN_BOOK_DEPTH_USD
                    trade_info['prediction'] = pred
                    
                    print(f"  >> 🚀 [{name}] Trade: {pred} | Filled ${trade_info['invested']:.2f} ({trade_info['shares']:.2f} shares) <<")
                    if trade_info['thin_book']:
                        print(f"  >> ⚠️ [{name}] Thin Book detected (Depth: ${fill['book_depth_usd']:.2f}) <<")
            
            commit_execution(pred_old, trade_old, "OLD")
            commit_execution(pred_trend, trade_trend, "TREND")
            
            traded_this_market = True
                
        # Wait 10 seconds before next poll
        time.sleep(10)

if __name__ == "__main__":
    continuous_live_pnl()
