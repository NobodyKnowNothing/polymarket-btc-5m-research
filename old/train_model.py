import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import pickle

def parse_features(features_json):
    if not features_json:
        return None
    
    if isinstance(features_json, str):
        try:
            data = json.loads(features_json)
        except:
            return None
    elif isinstance(features_json, list):
        data = features_json
    else:
        return None
        
    extracted = {}
    
    # Aggregates across exchanges
    obis = []
    deltas = []
    momentums = []
    volatilities = []
    funding_trends = []
    prices = []
    
    for ex_name in ['binance', 'coinbase', 'kraken']:
        extracted[f'{ex_name}_obi'] = np.nan
        extracted[f'{ex_name}_delta'] = np.nan
        extracted[f'{ex_name}_price'] = np.nan

    
    vol_map = {"low": 0, "normal": 1, "high": 2}
    
    for ex in data:
        ex_name = ex.get("exchange", "unknown")
        
        obi = ex.get("OBI", 0)
        delta = ex.get("Signed Delta", 0)
        
        if ex_name in ['binance', 'coinbase', 'kraken']:
            extracted[f'{ex_name}_obi'] = obi
            extracted[f'{ex_name}_delta'] = delta
        
        obis.append(obi)
        deltas.append(delta)
        
        # Prices
        if "current_price" in ex:
            prices.append(ex["current_price"])
            if ex_name in ['binance', 'coinbase', 'kraken']:
                extracted[f'{ex_name}_price'] = ex["current_price"]
            
        # Momentum
        mom_votes = ex.get("Momentum Votes", [])
        if mom_votes:
            momentums.append(sum(mom_votes))
            
        # Funding Trend
        fund_trend = ex.get("Funding Trend", [])
        if fund_trend:
            funding_trends.extend(fund_trend)
            
        # Volatility
        vol = ex.get("Volatility Regime", "normal")
        volatilities.append(vol_map.get(vol, 1))
        
    extracted['mean_obi'] = np.mean(obis) if obis else 0
    extracted['max_obi'] = np.max(obis) if obis else 0
    extracted['min_obi'] = np.min(obis) if obis else 0
    extracted['obi_spread'] = extracted['max_obi'] - extracted['min_obi']
    
    extracted['mean_delta'] = np.mean(deltas) if deltas else 0
    extracted['sum_delta'] = np.sum(deltas) if deltas else 0
    
    extracted['mean_momentum'] = np.mean(momentums) if momentums else 0
    extracted['sum_momentum'] = np.sum(momentums) if momentums else 0
    
    extracted['mean_volatility'] = np.mean(volatilities) if volatilities else 1
    extracted['max_volatility'] = np.max(volatilities) if volatilities else 1
    
    extracted['mean_funding'] = np.mean(funding_trends) if funding_trends else 0
    extracted['sum_funding'] = np.sum(funding_trends) if funding_trends else 0
    
    extracted['price_std'] = np.std(prices) if len(prices) > 1 else 0
    extracted['price_spread'] = np.max(prices) - np.min(prices) if len(prices) > 1 else 0
    
    extracted['binance_coinbase_price_spread'] = extracted['binance_price'] - extracted['coinbase_price']
    extracted['binance_kraken_price_spread'] = extracted['binance_price'] - extracted['kraken_price']
    
    extracted['obi_delta_interaction'] = extracted['mean_obi'] * extracted['mean_delta']
    extracted['momentum_delta_interaction'] = extracted['mean_momentum'] * extracted['mean_delta']
    extracted['funding_momentum_interaction'] = extracted['mean_funding'] * extracted['mean_momentum']
    extracted['volatility_obi_interaction'] = extracted['mean_volatility'] * extracted['mean_obi']
    
    return extracted

def main():
    conn = sqlite3.connect('backtest_data.db')
    
    query = """
    SELECT 
        s.market_slug, 
        s.timestamp,
        s.features, 
        s.seconds_remaining,
        o.actual_outcome
    FROM market_snapshots s
    JOIN market_outcomes o ON s.market_slug = o.market_slug
    WHERE o.actual_outcome IN ('Up', 'Down')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Sort by market and time so our sequential features make sense
    df = df.sort_values(by=['market_slug', 'timestamp'])
    
    parsed_features = df['features'].apply(parse_features)
    valid_idx = parsed_features.notna()
    df = df[valid_idx]
    parsed_features = parsed_features[valid_idx]
    
    X = pd.DataFrame(parsed_features.tolist(), index=df.index)
    
    # --- Time Series / Trend Feature Engineering ---
    # Since historical_collector runs every ~10s, 3 periods = 30s, 6 periods = 60s
    trend_cols = ['mean_obi', 'mean_delta', 'mean_momentum', 'binance_price', 'mean_funding', 'price_std']
    
    for col in trend_cols:
        if col in X.columns:
            # 30-second diff (3 intervals)
            X[f'{col}_diff_3'] = X.groupby(df['market_slug'])[col].diff(3).fillna(0)
            # 60-second diff (6 intervals)
            X[f'{col}_diff_6'] = X.groupby(df['market_slug'])[col].diff(6).fillna(0)
            # 60-second rolling mean
            X[f'{col}_roll_mean_6'] = X.groupby(df['market_slug'])[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
            # Rate of change over 60 seconds
            X[f'{col}_roc_6'] = X.groupby(df['market_slug'])[col].pct_change(periods=6).fillna(0).replace([np.inf, -np.inf], 0)

    X['seconds_remaining'] = df['seconds_remaining']
    
    y = df['actual_outcome'].map({'Up': 1, 'Down': 0})
    groups = df['market_slug']
    
    if len(X) < 10:
        print("Not enough data to train a model.")
        return
        
    unique_markets = sorted(df['market_slug'].unique())
    
    n_markets = len(unique_markets)
    train_end = int(n_markets * 0.6)
    val_end = int(n_markets * 0.8)
    
    train_markets = unique_markets[:train_end]
    val_markets = unique_markets[train_end:val_end]
    test_markets = unique_markets[val_end:]
    
    train_idx = df['market_slug'].isin(train_markets)
    val_idx = df['market_slug'].isin(val_markets)
    test_idx = df['market_slug'].isin(test_markets)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples, evaluating on {len(X_test)} samples.")
    
    clf = xgb.XGBClassifier(
        n_estimators=1000, 
        max_depth=2, 
        learning_rate=0.01, 
        subsample=0.5, 
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    print(f"Best iteration: {clf.best_iteration}")
    
    # Calibrate probabilities using the validation set
    print("Calibrating probabilities...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_val_prob_raw = clf.predict_proba(X_val)[:, 1]
    calibrator.fit(y_val_prob_raw, y_val)
    
    y_test_prob_raw = clf.predict_proba(X_test)[:, 1]
    y_prob = calibrator.predict(y_test_prob_raw)
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    print("\n--- Confidence Threshold Evaluation ---")
    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    print(f"{'Threshold':<12} | {'Coverage':<10} | {'Trades':<8} | {'Accuracy':<10}")
    print("-" * 50)
    for t in thresholds:
        high_conf_idx = (y_prob >= t) | (y_prob <= (1 - t))
        trades_taken = high_conf_idx.sum()
        if trades_taken > 0:
            y_pred_conf = (y_prob[high_conf_idx] >= 0.5).astype(int)
            y_test_conf = y_test.iloc[high_conf_idx]
            acc = accuracy_score(y_test_conf, y_pred_conf)
            coverage = trades_taken / len(y_test)
            print(f"{t:<12.2f} | {coverage:<10.1%} | {trades_taken:<8} | {acc:<10.4f}")
        else:
            print(f"{t:<12.2f} | {'0.0%':<10} | {0:<8} | {'N/A':<10}")
    
    print("\nFeature Importances:")
    importances = clf.feature_importances_
    
    # Sort importances
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:20}: {importances[idx]:.4f}")
        
    print("\nSaving calibrated model to trend_model.pkl...")
    with open("trend_model.pkl", "wb") as f:
        pickle.dump({'model': clf, 'calibrator': calibrator, 'features': list(X.columns)}, f)
    
    # Also save the base XGB model just in case other scripts need the raw JSON
    clf.save_model("trend_xgb_model.json")
    print("Done!")

if __name__ == "__main__":
    main()
