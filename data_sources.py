"""
Data Sources — READ-ONLY module (analogous to autoresearch's prepare.py).

Fetches live BTC market data from multiple exchanges (Binance, Coinbase,
Kraken) and returns a list of feature dictionaries that strategies consume.

Features per exchange:
    - OBI           (Order Book Imbalance, [-1, 1])
    - Signed Delta  (net trade flow over last 90s)
    - Funding Trend (list of recent perp funding rates)
    - Volatility Regime  ("high" or "low")
    - Momentum Votes     (list of [-1, 0, 1] for last 3 candle closes)

DO NOT MODIFY THIS FILE. Strategies should treat the feature schema as fixed.
"""

import ccxt
import numpy as np
import pandas as pd


def get_exchange_data(exchange_id: str, symbol: str = "BTC/USDT") -> dict:
    """Fetch a full feature snapshot from a single exchange."""
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()

    # Resolve symbol
    target_symbol = symbol
    if symbol not in exchange.symbols:
        for alt in ["BTC/USD", "BTC/USDC", "BTC-USD"]:
            if alt in exchange.symbols:
                target_symbol = alt
                break

    # 1. L2 Order Book → OBI
    ob = exchange.fetch_l2_order_book(target_symbol)
    bid_vol = sum(b[1] for b in ob["bids"][:10])
    ask_vol = sum(a[1] for a in ob["asks"][:10])
    obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) else 0
    current_price = (ob["bids"][0][0] + ob["asks"][0][0]) / 2

    # 2. Last 90s trades → signed delta
    now = exchange.milliseconds()
    since = now - 90_000
    params = {"until": now} if exchange_id == "coinbase" else {}
    trades = exchange.fetch_trades(target_symbol, since=since, params=params)
    df_trades = pd.DataFrame(trades)
    signed_delta = 0.0
    if not df_trades.empty:
        df_trades["signed_v"] = df_trades.apply(
            lambda x: x["amount"] if x["side"] == "buy" else -x["amount"], axis=1
        )
        signed_delta = float(df_trades["signed_v"].sum())

    # 3. Funding rate trend (perps)
    funding_trend = []
    swap_symbol = target_symbol + ":USDT" if ":USDT" not in target_symbol else target_symbol
    if swap_symbol not in exchange.symbols:
        for s in [s for s in exchange.symbols if "BTC" in s and ":" in s]:
            swap_symbol = s
            break
    try:
        if exchange.has.get("fetchFundingRateHistory"):
            funding = exchange.fetch_funding_rate_history(swap_symbol, limit=4)
            funding_trend = [f["fundingRate"] for f in funding]
    except Exception:
        pass

    # 4. Volatility regime + momentum
    ohlcv = exchange.fetch_ohlcv(target_symbol, timeframe="1m", limit=20)
    df_ohlcv = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    returns = np.log(df_ohlcv["c"] / df_ohlcv["c"].shift(1))
    volatility = float(returns.std())
    regime = "high" if volatility > 0.0005 else "low"

    last_3 = df_ohlcv["c"].tail(3).values
    momentum_votes = np.sign(np.diff(last_3)).tolist()

    return {
        "exchange": exchange_id,
        "symbol": target_symbol,
        "swap_symbol": swap_symbol if funding_trend else "N/A",
        "current_price": round(current_price, 2),
        "OBI": round(obi, 4),
        "Signed Delta": round(signed_delta, 4),
        "Funding Trend": funding_trend,
        "Volatility Regime": regime,
        "Momentum Votes": momentum_votes,
    }


def get_btc_features() -> list[dict]:
    """
    Fetch features from all configured exchanges.
    Returns a list of feature dicts (one per exchange).
    Silently skips exchanges that error.
    """
    exchanges = ["binance", "coinbase", "kraken"]
    results = []
    for ex in exchanges:
        try:
            results.append(get_exchange_data(ex))
        except Exception as e:
            print(f"[data_sources] Error fetching {ex}: {e}")
    return results
