import sqlite3
import json
import numpy as np
import pandas as pd
import os

def parse_features(features_json):
    """Parses the nested exchange JSON features into a flat dictionary."""
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
    
    # Cross-exchange spreads
    if not np.isnan(extracted['binance_price']) and not np.isnan(extracted['coinbase_price']):
        extracted['binance_coinbase_price_spread'] = extracted['binance_price'] - extracted['coinbase_price']
    else:
        extracted['binance_coinbase_price_spread'] = 0
        
    if not np.isnan(extracted['binance_price']) and not np.isnan(extracted['kraken_price']):
        extracted['binance_kraken_price_spread'] = extracted['binance_price'] - extracted['kraken_price']
    else:
        extracted['binance_kraken_price_spread'] = 0
    
    # Simple interactions
    extracted['obi_delta_interaction'] = extracted['mean_obi'] * extracted['mean_delta']
    extracted['momentum_delta_interaction'] = extracted['mean_momentum'] * extracted['mean_delta']
    extracted['funding_momentum_interaction'] = extracted['mean_funding'] * extracted['mean_momentum']
    extracted['volatility_obi_interaction'] = extracted['mean_volatility'] * extracted['mean_obi']
    
    return extracted

def load_raw_dataset(db_path='backtest_data.db'):
    """Loads raw snapshots and outcomes from SQLite and returns a flat DataFrame."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
        
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        s.market_slug, 
        s.timestamp,
        s.features,
        s.prices,
        s.seconds_remaining,
        o.actual_outcome
    FROM market_snapshots s
    JOIN market_outcomes o ON s.market_slug = o.market_slug
    WHERE o.actual_outcome IN ('Up', 'Down')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Sort by market and time
    df = df.sort_values(by=['market_slug', 'timestamp'])
    
    # Parse JSON features
    parsed = df['features'].apply(parse_features)

    # Parse Polymarket prices
    def parse_prices(prices_str):
        if not prices_str: return None
        try:
            return json.loads(prices_str)
        except:
            return None

    prices_json = df['prices'].apply(parse_prices)
    pm_cost_up = prices_json.apply(lambda x: x.get('Up', {}).get('best_ask', 0.5) if x else 0.5)
    pm_cost_down = prices_json.apply(lambda x: x.get('Down', {}).get('best_ask', 0.5) if x else 0.5)
    
    valid_idx = parsed.notna()
    df = df[valid_idx]
    parsed = parsed[valid_idx]
    pm_cost_up = pm_cost_up[valid_idx]
    pm_cost_down = pm_cost_down[valid_idx]
    
    # Convert list of dicts to DataFrame columns
    feature_df = pd.DataFrame(parsed.tolist(), index=df.index)
    
    # Merge back with metadata
    final_df = pd.concat([
        df[['market_slug', 'timestamp', 'seconds_remaining', 'actual_outcome']],
        pm_cost_up.rename('pm_cost_up'),
        pm_cost_down.rename('pm_cost_down'),
        feature_df
    ], axis=1)
    
    # FIX 1: Capture the strike price and calculate price vs strike features
    # The first observed binance_price for each market serves as the initial strike price
    first_prices = final_df.groupby('market_slug')['binance_price'].transform('first')
    final_df['strike_price'] = first_prices
    final_df['price_vs_strike'] = final_df['binance_price'] - final_df['strike_price']
    final_df['price_vs_strike_pct'] = (final_df['binance_price'] - final_df['strike_price']) / final_df['strike_price']
    
    # Filter for seconds_remaining > 140 to match early live inference (150s)
    final_df = final_df[final_df['seconds_remaining'] > 140]
    
    return final_df
