import asyncio
import sqlite3
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Adjust path to import local modules
sys.path.insert(0, str(Path(__file__).parent))
try:
    from main import PolymarketBTC5m
    from data_sources import get_btc_features
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

DB_PATH = Path(__file__).parent / "backtest_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            market_slug TEXT,
            features JSON,
            prices JSON,
            seconds_remaining REAL,
            actual_outcome TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_outcomes (
            market_slug TEXT PRIMARY KEY,
            actual_outcome TEXT,
            resolution_mids JSON
        )
    """)
    conn.commit()
    conn.close()

async def collect_snapshot(market, current_slug, conn):
    """Fetch features and prices, save to DB."""
    try:
        features = await asyncio.to_thread(get_btc_features)
        prices_state = await asyncio.to_thread(market.get_prices)
        
        status = prices_state.get("status")
        
        if status == "expired":
            # The market advanced internally during get_prices()
            # prices_state["outcome"] has the final orderbook for the old market,
            # but we don't save a normal snapshot for it. We return the new slug.
            return prices_state.get("new_slug")
            
        prices = prices_state.get("prices", {})
        seconds_remaining = max(0.0, market.market_end_time - time.time())
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO market_snapshots (timestamp, market_slug, features, prices, seconds_remaining)
            VALUES (?, ?, ?, ?, ?)
        """, (
            time.time(),
            current_slug,
            json.dumps(features),
            json.dumps(prices),
            seconds_remaining
        ))
        conn.commit()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved snapshot for {current_slug} ({seconds_remaining:.0f}s left)")
        return current_slug
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error collecting snapshot: {e}")
        return current_slug

async def resolve_market(slug, conn):
    """Wait and resolve an old market, updating its outcome."""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Awaiting resolution for {slug}...")
        # Give it a bit of time to settle on the blockchain/gamma api
        await asyncio.sleep(10)
        
        timestamp = int(slug.split("-")[-1])
        market = await asyncio.to_thread(PolymarketBTC5m, initial_ts=timestamp)
        state = await asyncio.to_thread(market.get_prices)
        
        prices = state.get("prices") or state.get("outcome", {})
        up_mid = prices.get("Up", {}).get("midpoint", 0.0)
        down_mid = prices.get("Down", {}).get("midpoint", 0.0)
        
        outcome = None
        if up_mid > down_mid:
            outcome = "Up"
        elif down_mid > up_mid:
            outcome = "Down"
            
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_outcomes (market_slug, actual_outcome, resolution_mids)
            VALUES (?, ?, ?)
        """, (slug, outcome, json.dumps({"Up": up_mid, "Down": down_mid})))
        
        # Also update snapshots for convenience
        cursor.execute("""
            UPDATE market_snapshots SET actual_outcome = ? WHERE market_slug = ?
        """, (outcome, slug))
        
        conn.commit()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Resolved {slug} -> {outcome}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error resolving market {slug}: {e}")

async def main_loop():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    current_market = None
    current_slug = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Historical data collector started. Saving to {DB_PATH}")
    
    while True:
        try:
            if current_market is None or current_market.is_expired:
                if current_slug is not None:
                    # Spawn task to resolve the old market
                    asyncio.create_task(resolve_market(current_slug, conn))
                
                # Init new market
                current_market = await asyncio.to_thread(PolymarketBTC5m)
                state = await asyncio.to_thread(current_market.get_prices)
                current_slug = state.get("slug", "unknown")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Tracking new market: {current_slug}")
            
            # Collect snapshot
            new_slug = await collect_snapshot(current_market, current_slug, conn)
            
            if new_slug and new_slug != current_slug:
                # The market auto-advanced inside collect_snapshot!
                # We need to spawn a resolver for the old slug
                asyncio.create_task(resolve_market(current_slug, conn))
                current_slug = new_slug
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Tracking new market: {current_slug}")
            
            # Wait before next snapshot (e.g., 10 seconds)
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Main loop error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Collector stopped.")
