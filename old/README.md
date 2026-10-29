# Polymarket Degen Exploiter 9000

## Overview
This project is an autonomous quantitative trading research pipeline. It continuously evolves and tests trading strategies for Polymarket's 5-minute Bitcoin (BTC) Up/Down market. The core innovation is using an LLM (Gemma-4-26b) as the "mutation operator" in an evolutionary algorithm. By feeding the LLM historical performance and live market data, it iteratively improves a Python-based trading strategy.

## How it Works
The pipeline operates on a continuous 5-minute cycle matching the Polymarket market resolution:

1. **Feature Extraction**: It pulls live data from Binance, Coinbase, and Kraken (`data_sources.py`). This includes:
   - Order Book Imbalance (OBI)
   - Signed Delta (aggressive trade flow)
   - Funding Rate Trends
   - Volatility Regimes
   - Momentum
2. **Strategy Generation**: The LLM agent reads the current best strategy, the new market features, and a history of what worked in the past. It then generates multiple mutated strategies (variants) per generation (`autoresearch.py`).
3. **Execution & Simulation**: Each variant outputs a prediction ("Up" or "Down") and a bet amount. 
4. **Market Resolution & Scoring**: The system waits for the 5-minute window to close and checks the actual outcome. It then scores the variants using Return on Investment (ROI). To ensure realism, a paper trading module (`paper_trade.py`) factors in Polymarket's taker fees and orderbook slippage.
5. **Git Ratchet**: The strategy with the highest positive ROI is declared the winner. It is committed to a local git repository (`agent_repo`) and serves as the baseline strategy for the next generation. Failed mutations are discarded.

## Key Files
- `autoresearch.py`: The main loop. Orchestrates LLM calls, executes sandboxed strategy code, manages the git ratchet, and tracks results.
- `main.py`: Manages the Polymarket API interaction. Provides the `PolymarketBTC5m` class which fetches live orderbooks, calculates midpoint pricing, and tracks expiration.
- `data_sources.py`: Connects to crypto exchanges via CCXT to aggregate L2 and trade data into clean features.
- `paper_trade.py`: A realistic position-sizing calculator that walks the Polymarket orderbook to compute precise slippage and fees.
- `program.md`: The system prompt for the LLM, containing the context, rules, and strategy design tips.
- `results.tsv`: Tracks the historical performance (predictions, outcomes, profit, ROI) of all strategy mutations.
- `strategy.py` (inside `agent_repo/`): The mutable strategy logic actively being rewritten by the LLM.

## Usage
First, ensure you have your `GOOGLE_API_KEY` set in a `.env` file in the root directory:
```env
GOOGLE_API_KEY='your_api_key_here'
```

To run the evolutionary pipeline indefinitely:
```bash
source venv/bin/activate
python autoresearch.py
```
Press `Ctrl+C` to terminate the loop safely.
