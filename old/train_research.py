import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.isotonic import IsotonicRegression
from train_data import load_raw_dataset

"""
KARPATHY-STYLE AUTORESEARCH TEMPLATE
This script is designed to be modified by an LLM to find better trading models.
The data loading is handled by train_data.py.
Focus your research on:
1. Feature Engineering (adding/modifying trend features)
2. Model Hyperparameters
3. Training Strategy (regularization, early stopping)
4. Calibration and Thresholding
"""

def engineer_features(df):
    """
    Apply time-series / trend feature engineering.
    Modify this function to try different feature sets.
    """
    X = df.drop(columns=['market_slug', 'timestamp', 'actual_outcome'])
    
    # Since historical_collector runs every ~10s, 3 periods = 30s, 6 periods = 60s
    trend_cols = ['mean_obi', 'mean_delta', 'mean_momentum', 'binance_price', 'mean_funding', 'price_std']
    
    for col in trend_cols:
        if col in X.columns:
            # 30-second diff (3 intervals)
            X[f'{col}_diff_3'] = df.groupby('market_slug')[col].diff(3).fillna(0)
            # 60-second diff (6 intervals)
            X[f'{col}_diff_6'] = df.groupby('market_slug')[col].diff(6).fillna(0)
            # 60-second rolling mean
            X[f'{col}_roll_mean_6'] = df.groupby('market_slug')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
            # Rate of change over 60 seconds
            X[f'{col}_roc_6'] = df.groupby('market_slug')[col].pct_change(periods=6).fillna(0).replace([np.inf, -np.inf], 0)

    return X

def train_and_evaluate():
    print("Loading raw data...")
    df = load_raw_dataset()
    
    if len(df) < 10:
        print("Not enough data to train a model.")
        return
        
    print(f"Loaded {len(df)} samples across {df['market_slug'].nunique()} markets.")

    # 1. Feature Engineering
    print("Engineering features...")
    X = engineer_features(df)
    y = df['actual_outcome'].map({'Up': 1, 'Down': 0})
    
    # 2. Splitting by market (to avoid data leakage)
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
    
    print(f"Training: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 3. Model Definition and Training
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
    
    # 4. Calibration
    print("Calibrating probabilities...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_val_prob_raw = clf.predict_proba(X_val)[:, 1]
    calibrator.fit(y_val_prob_raw, y_val)
    
    # 5. Evaluation
    y_test_prob_raw = clf.predict_proba(X_test)[:, 1]
    y_prob = calibrator.predict(y_test_prob_raw)
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    # Confidence threshold analysis
    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75]
    print(f"\n{'Threshold':<10} | {'Coverage':<10} | {'Accuracy':<10}")
    print("-" * 35)
    for t in thresholds:
        high_conf_idx = (y_prob >= t) | (y_prob <= (1 - t))
        trades = high_conf_idx.sum()
        if trades > 0:
            acc_conf = accuracy_score(y_test[high_conf_idx], (y_prob[high_conf_idx] >= 0.5).astype(int))
            print(f"{t:<10.2f} | {trades/len(y_test):<10.1%} | {acc_conf:<10.4f}")
            
    # 6. Save results
    save_path = "trend_model.pkl"
    print(f"\nSaving model to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump({'model': clf, 'calibrator': calibrator, 'features': list(X.columns)}, f)
    
    # Also save metadata for the search loop to read
    results = {
        "accuracy": float(acc),
        "best_iteration": int(clf.best_iteration),
        "n_samples": len(df),
        "feature_count": len(X.columns)
    }
    import json
    with open("research_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    train_and_evaluate()
