import time
import xgboost as xgb
import pandas as pd
import warnings
from main import PolymarketBTC5m
from data_sources import get_btc_features
from train_model import parse_features

warnings.filterwarnings("ignore")

def test_model_live():
    print("Loading XGBoost model & Calibrator...")
    import pickle
    with open("final_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
        
    clf = saved_data['model']
    calibrator = saved_data['calibrator']
    expected_features = saved_data['features']
    
    market = PolymarketBTC5m()
    secs_remaining = market.market_end_time - time.time()
    
    # User asked for the *next full market window*
    # If the current one is already midway, wait for the next one to start
    if secs_remaining < 270: # less than 4.5 mins left
        print(f"Current market {market.slug} has {secs_remaining:.0f}s left.")
        print("Waiting for the next full market window to begin...")
        while not market.is_expired:
            rem = market.market_end_time - time.time()
            if rem > 0:
                time.sleep(min(10, rem + 0.1))
        
        # Advance
        market.get_prices()
        market = PolymarketBTC5m()
        
    print(f"\n--- Testing on Full Market Window: {market.slug} ---")
    
    print("Fetching live features...")
    raw_features = get_btc_features()
    parsed = parse_features(raw_features)
    
    if not parsed:
        print("Failed to fetch features.")
        return
        
    X_test = pd.DataFrame([parsed])
    X_test['seconds_remaining'] = max(0, market.market_end_time - time.time())
    
    # Ensure all expected columns are present
    for col in expected_features:
        if col not in X_test.columns:
            X_test[col] = 0.0
            
    # Order columns exactly as expected
    X_test = X_test[expected_features]
    
    y_prob_raw = clf.predict_proba(X_test)[:, 1]
    true_prob_up = calibrator.predict(y_prob_raw)[0]
    true_prob_down = 1.0 - true_prob_up
    
    print(f"Calibrated True Probability (Up): {true_prob_up:.4f}")
    
    # Fetch live Polymarket orderbook for EV calculation
    market_data = market.get_prices()
    prices = market_data.get("prices", {})
    up_ask = prices.get("Up", {}).get("best_ask", 1.0) # default 1.0 prevents buying if missing
    down_ask = prices.get("Down", {}).get("best_ask", 1.0)
    
    # Expected Value = Probability of Winning - Cost of Share
    ev_up = true_prob_up - up_ask
    ev_down = true_prob_down - down_ask
    
    print(f"Up Ask:   {up_ask:<6.4f} | EV Up:   {ev_up:.4f}")
    print(f"Down Ask: {down_ask:<6.4f} | EV Down: {ev_down:.4f}")
    
    # EV Margin required to enter a trade
    MIN_EV_MARGIN = 0.03
    
    prediction = None
    if ev_up > ev_down and ev_up > MIN_EV_MARGIN:
        prediction = "Up"
    elif ev_down > ev_up and ev_down > MIN_EV_MARGIN:
        prediction = "Down"
        
    print(f"Model Probability (Up): {true_prob_up:.4f}")
    if prediction:
        print(f"🚀 High EV trade found! Taking position: {prediction}!")
    else:
        print("⏸️ Model abstain (EV too low or negative).")
        
    secs_to_wait = max(0, market.market_end_time - time.time())
    print(f"\nWaiting for market resolution ({secs_to_wait:.0f}s left)...")
    
    while not market.is_expired:
        rem = market.market_end_time - time.time()
        if rem > 0:
            time.sleep(min(15, rem + 1))
            
    print("\nMarket expired! Resolving...")
    time.sleep(5) # wait for oracle
    
    result = market.get_prices()
    prices = result.get("outcome", {})
    up_mid = prices.get("Up", {}).get("midpoint", 0.0)
    down_mid = prices.get("Down", {}).get("midpoint", 0.0)
    
    actual = None
    if up_mid > down_mid:
        actual = "Up"
    elif down_mid > up_mid:
        actual = "Down"
        
    print(f"Actual Outcome: {actual} (Up mid: {up_mid}, Down mid: {down_mid})")
    
    if prediction:
        if prediction == actual:
            print("✅ WIN!")
        else:
            print("❌ LOSS!")
    else:
        print("No trade taken.")

if __name__ == "__main__":
    test_model_live()
