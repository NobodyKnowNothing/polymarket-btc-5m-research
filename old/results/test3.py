import sys
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

import agent_repo.strategy as s

def debug_predict():
    features = get_btc_features()
    if not features:
        print("NO FEATURES")
        return None

    weights = {"Binance": 0.5, "Coinbase": 0.3, "Kraken": 0.2}
    total_obi, total_delta, total_mom, total_funding, total_w = 0.0, 0.0, 0.0, 0.0, 0.0
    divergence_count = 0
    exchanges_with_data = 0
    import numpy as np
    for f in features:
        w = weights.get(f.get("exchange", ""), 0.25)
        exchanges_with_data += 1
        obi = f.get("OBI", 0.0)
        delta = np.clip(f.get("Signed Delta", 0.0) / 1000.0, -1.0, 1.0)
        mom_list = f.get("Momentum Votes", [])
        mom = np.mean(mom_list) if mom_list else 0.0
        
        fund_list = f.get("Funding Trend", [])
        funding = np.mean(fund_list) if fund_list else 0.0

        if (obi > 0.05 and delta < -0.05) or (obi < -0.05 and delta > 0.05):
            divergence_count += 1

        total_obi += obi * w
        total_delta += delta * w
        total_mom += mom * w
        total_funding += funding * w
        total_w += w

    if total_w == 0:
        print("TOTAL W 0")
        return None

    avg_obi = total_obi / total_w
    avg_delta = total_delta / total_w
    avg_mom = total_mom / total_w
    avg_funding = total_funding / total_w

    pressure_signal = (avg_obi * 0.45) + (avg_delta * 0.45) + (avg_mom * 0.1)
    if divergence_count > 0:
        pressure_signal *= (1.0 - (0.5 * (divergence_count / exchanges_with_data)))
    
    pressure_signal -= (avg_funding * 2.0)
    prob_up = np.clip((pressure_signal + 1.0) / 2.0, 0.0, 1.0)

    m_prices = get_market_prices()
    if not m_prices or "Up" not in m_prices or "Down" not in m_prices:
        print("NO PRICES")
        return None

    up_price = m_prices["Up"].get("best_ask", m_prices["Up"].get("midpoint", 0.5))
    down_price = m_prices["Down"].get("best_ask", m_prices["Down"].get("midpoint", 0.5))

    is_high_vol = any(f.get("Volatility Regime") == "high" for f in features)
    margin = 0.04 if is_high_vol else 0.02

    edge_up = prob_up - up_price
    edge_down = (1.0 - prob_up) - down_price
    
    print(f"avg_obi: {avg_obi}, avg_delta: {avg_delta}, avg_mom: {avg_mom}, avg_funding: {avg_funding}")
    print(f"pressure_signal: {pressure_signal}, prob_up: {prob_up}")
    print(f"up_price: {up_price}, down_price: {down_price}, margin: {margin}")
    print(f"edge_up: {edge_up}, edge_down: {edge_down}")

debug_predict()
