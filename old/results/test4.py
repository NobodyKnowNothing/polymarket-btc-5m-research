import sys
sys.path.insert(0, "/home/owenwalker/Projects/polymarketdegenexploiter9k")
from main import PolymarketBTC5m
market = PolymarketBTC5m()
print(market.get_prices())
