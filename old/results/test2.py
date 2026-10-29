import sys
from pathlib import Path
import time
sys.path.insert(0, "/home/owenwalker/Projects/polymarketdegenexploiter9k")
from data_sources import get_btc_features
from main import PolymarketBTC5m
market = PolymarketBTC5m()
print(market.get_prices())
