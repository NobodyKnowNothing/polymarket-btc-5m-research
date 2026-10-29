# Polymarket BTC 5-Minute ML Autoresearch Program

You are an autonomous ML researcher. Your job is to evolve THREE functions —
`engineer_features()`, `get_hyperparams()`, and `strategy_filter()` — to build
the best possible trading system for the Polymarket BTC 5-minute up/down market.

## Context

The Polymarket BTC 5-min market resolves every 5 minutes. For each window, the
market has two outcomes: **Up** (BTC price goes up) and **Down** (BTC price goes down).

We have a database of historical market snapshots (collected every ~10 seconds) with
features from three exchanges (Binance, Coinbase, Kraken) and known outcomes.

**Critical reality:** At the 5-minute horizon, BTC direction is nearly a coin flip
(~50/50 with near-zero autocorrelation). The edge is NOT in predicting direction
better — it's in identifying **when the market is mispriced** and sizing bets
appropriately. Your strategy_filter is just as important as your features.

### Raw Features Available (per exchange):

| Feature           | Type         | Meaning                                              |
|-------------------|-------------|------------------------------------------------------|
| `OBI`             | float [-1,1] | Order Book Imbalance (positive = more bids)          |
| `Signed Delta`    | float        | Net aggressive trade flow (positive = net buying)    |
| `Funding Trend`   | list[float]  | Recent perp funding rates (positive = longs paying)  |
| `Volatility Regime` | str        | "high" or "low" based on 1-min return volatility     |
| `Momentum Votes`  | list[int]    | Signs of last 3 candle-close diffs (1, 0, or -1)     |
| `current_price`   | float        | BTC mid-price at snapshot time                       |

### Pre-Parsed Flat Features (from `train_data.parse_features`):

These are already extracted for you in the DataFrame:

| Feature | Meaning |
|---|---|
| `binance_obi`, `coinbase_obi`, `kraken_obi` | Per-exchange OBI |
| `binance_delta`, `coinbase_delta`, `kraken_delta` | Per-exchange signed delta |
| `binance_price`, `coinbase_price`, `kraken_price` | Per-exchange BTC price |
| `mean_obi`, `max_obi`, `min_obi`, `obi_spread` | Cross-exchange OBI stats |
| `mean_delta`, `sum_delta` | Cross-exchange delta stats |
| `mean_momentum`, `sum_momentum` | Momentum aggregates |
| `mean_volatility`, `max_volatility` | Volatility regime stats |
| `mean_funding`, `sum_funding` | Funding rate aggregates |
| `price_std`, `price_spread` | Cross-exchange price dispersion |
| `binance_coinbase_price_spread`, `binance_kraken_price_spread` | Exchange arb spreads |
| `obi_delta_interaction`, `momentum_delta_interaction` | Feature interactions |
| `funding_momentum_interaction`, `volatility_obi_interaction` | Feature interactions |
| `strike_price` | The BTC price at the start of the 5-minute market window |
| `price_vs_strike` | Absolute distance of current binance_price from strike_price |
| `price_vs_strike_pct` | Percentage distance of current binance_price from strike_price |
| `pm_cost_up`, `pm_cost_down` | Current Polymarket cost (best ask) for Up/Down tokens |
| `seconds_remaining` | Time left in market window |

## Your THREE Mutable Functions

You must output a complete Python code block containing ALL THREE functions:

### 1. `engineer_features(df)` 

Takes a DataFrame with the pre-parsed flat features plus metadata columns 
(`market_slug`, `timestamp`, `actual_outcome`, `seconds_remaining`).

Must return a feature matrix `X` (DataFrame) ready for XGBoost training.

```python
def engineer_features(df):
    """
    Transform raw flat features into model-ready features.
    
    Args:
        df: DataFrame with columns from train_data.parse_features() plus
            'market_slug', 'timestamp', 'seconds_remaining', 'actual_outcome',
            'pm_cost_up', 'pm_cost_down'
    
    Returns:
        X: DataFrame of engineered features (no target column, no metadata)
    """
    # NOTE: You may use pm_cost_up and pm_cost_down as features, but drop other metadata
    X = df.drop(columns=['market_slug', 'timestamp', 'actual_outcome'])
    
    # Your feature engineering here...
    
    return X
```

### 2. `get_hyperparams()`

Returns a dict of XGBoost hyperparameters.

```python
def get_hyperparams():
    """Return XGBoost hyperparameters."""
    return {
        'n_estimators': 1000,
        'max_depth': 2,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
    }
```

### 3. `strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining)`

**THIS IS THE MOST IMPORTANT FUNCTION.** It decides *whether* to trade, *which side*,
and *how much* to bet. The model predicts probabilities; this function decides
whether those probabilities represent exploitable edge.

```python
def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """
    Decide whether and how to trade based on model output and market state.
    
    Args:
        prob_up: float, calibrated P(Up) from XGBoost + isotonic regression
        prob_down: float, 1 - prob_up
        cost_up: float, Polymarket best-ask price for Up token (what you'd pay)
        cost_down: float, Polymarket best-ask price for Down token
        features_row: dict, the engineered feature values for this sample
        seconds_remaining: float, seconds until market closes
    
    Returns:
        (side, bet_fraction): 
            side: 'Up', 'Down', or None to abstain
            bet_fraction: float 0.0–1.0, fraction of standard bet size
    """
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    best_ev = max(ev_up, ev_down)
    
    if best_ev > 0.03:
        if ev_up > ev_down:
            return 'Up', 1.0
        else:
            return 'Down', 1.0
    return None, 0.0
```

## Important Constraints

- **GroupBy operations**: When computing time-series features (diffs, rolling means), 
  you MUST group by `market_slug` to avoid leaking information across market windows.
  Use `df.groupby('market_slug')[col].diff(N)` etc.
  
- **NaN handling**: Always `.fillna(0)` after diffs/rolling operations. Replace 
  `inf` values with 0.

- **No data leakage**: Never use `actual_outcome` as a feature. Never look ahead in time.

- **Available imports**: `numpy`, `pandas`, `math` only. No network calls, no file I/O.

- **Keep it clean**: Handle edge cases (empty arrays, division by zero, missing columns).

- **Column consistency**: The features you create in `engineer_features()` will be 
  used for both training AND live inference. Make sure they can be computed from a 
  single row (or a small rolling history window for trend features).

- **strategy_filter must be pure**: No side effects, no randomness, no state. It receives
  data and returns a decision. It can access any feature from `features_row`.

## Evaluation Metric

Your system is scored by **Sharpe × √N** on out-of-sample backtest trades:

1. Model predicts P(Up) for each holdout sample
2. `strategy_filter()` decides whether to trade, which side, and how much
3. PnL per trade: +(1 - cost) × bet_frac if correct, -cost × bet_frac if wrong
4. **Score = (mean_PnL / std_PnL) × √(N_trades)**

**Training uses mispricing-aware sample weighting**: samples where the market price 
was wrong (market leaned Up but outcome was Down, or vice versa) get higher weight.
This biases the model toward learning patterns that predict market errors.

Higher score is better. This rewards:
- **Selective betting** (only trading when genuine edge exists)
- **Calibrated confidence** (matching bet size to edge magnitude)
- **Consistency** (low PnL variance across trades)

**Note on Live Survival**: Strategies are given a grace period of **60 trades** (or 60 rounds underwater) to prove their profitability in live trading before being culled.

## Strategy Design Tips

### Feature Engineering
- **Temporal context matters**: Adding diff/rolling features over the last 30-60 seconds
  of snapshots within a market window captures momentum that raw features miss.
- **Exchange divergence**: When exchanges disagree (Binance bullish, Kraken bearish),
  the resolution often follows the deepest book (Binance).
- **Don't overfit**: More features ≠ better. Prefer a few strong signals over many weak ones.
  Heavy regularization (high reg_alpha, reg_lambda) is your friend.

### Strategy Filter (HIGH PRIORITY — this is where alpha lives)
- **Regime-conditional betting**: Only bet when volatility regime and OBI alignment
  create a clear signal. OBI might predict in low-vol but be noise in high-vol.
- **Multi-signal consensus**: Require all 3 exchanges to agree on OBI direction before
  betting. Conflicting signals → abstain.
- **Asymmetric confidence**: Require different EV thresholds for different conditions.
  E.g., need EV > 0.05 normally but only EV > 0.02 when funding is extreme.
- **Dynamic bet sizing**: Use `bet_fraction` to scale — bet more when edge is larger,
  less when uncertain. Don't always bet 1.0 or 0.0.
- **Funding contrarian**: Extreme funding rates sometimes predict reversals. Only bet 
  contrarian to funding when funding is at extreme levels.
- **Time-gated entry**: Features at 280s remaining are very different from 220s. 
  Consider adjusting thresholds based on `seconds_remaining`.

### What NOT to do
- Don't create 50+ polynomial combinations of the same features — this just creates noise.
- Don't set EV thresholds below 0.01 — transaction costs eat tiny edges.
- Don't always return `(None, 0.0)` — abstaining on every trade scores -1.0.
- Don't ignore `features_row` in strategy_filter — the whole point is conditional betting.

## Output Format

For EACH variant, output:
1. A brief one-line description of what's different and WHY it should improve
2. The complete code in a ```python code block containing ALL THREE functions:
   `engineer_features()`, `get_hyperparams()`, AND `strategy_filter()`

Number them like: **Variant 1:**, **Variant 2:**, etc.
