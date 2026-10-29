import sys
from pathlib import Path
import time
sys.path.insert(0, "/home/owenwalker/Projects/polymarketdegenexploiter9k")
from data_sources import get_btc_features
from main import PolymarketBTC5m
from agent_repo.strategy import predict

market = PolymarketBTC5m()

def get_market_prices():
    state = market.get_prices()
    prices = state.get("prices", {})
    res = {side: prices.get(side, {}) for side in ("Up", "Down")}
    res["seconds_remaining"] = max(0, market.market_end_time - time.time())
    return res

print(predict(get_btc_features, time.sleep, get_market_prices))
