#!/usr/bin/env python3
"""
Unified ML Autoresearch Loop for Polymarket BTC 5-min Trading.

Integrates: train_data, train_model, train_research, test_live_model, autoresearch, data_sources

Loop:
  1. LLM mutates feature engineering code + hyperparams
  2. Train XGBoost on historical data (60% train, 20% val)
  3. Backtest on holdout (20% test) — score = Sharpe * sqrt(N)
  4. Winners promoted to live Polymarket testing
  5. Live results fed back to LLM for next iteration
"""

import json, math, os, re, sys, textwrap, time, traceback, pickle, warnings, threading, subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from google import genai
from google.genai import types
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# --- Config ---
MODEL = "gemma-4-26b-a4b-it"
MAX_POPULATION = 200
MIN_BOOK_DEPTH_USD = 10.0
BET_SIZE = 1.0
EV_MARGIN = 0.03
BACKTEST_SCORE_THRESHOLD = 1.5  # min Sharpe*sqrt(N) to go live (p < 0.05)
RETRAIN_INTERVAL = 20            # retrain promoted models every N generations
RETRAIN_EMA_ALPHA = 0.3          # EMA smoothing for backtest score updates
MAX_UNDERWATER_ROUNDS = 60       # how many rounds a strategy's score/PnL can be negative/zero before it is thrown out
MAX_UNPROFITABLE_TRADES = 60     # how many live trades before an unprofitable strategy (PnL <= 0) is thrown out
MAX_ABSTAIN_ROUNDS = 60          # how many consecutive rounds a strategy can abstain from trading before it is thrown out
RESULTS_FILE = Path(__file__).parent / "ml_results.tsv"
PROGRAM_FILE = Path(__file__).parent / "program_ml.md"
MODELS_DIR = Path(__file__).parent / "ml_models"
STRATEGIES_DIR = Path(__file__).parent / "strategies"
STRATEGY_FILE = Path(__file__).parent / "best_strategy.py"
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

MODELS_DIR.mkdir(exist_ok=True)
STRATEGIES_DIR.mkdir(exist_ok=True)

def log(msg):
    timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(timestamped, flush=True)
    try:
        # Master log
        with open("pipeline.log", "a") as f:
            f.write(timestamped + "\n")
        # Split logs
        if "[LIVE]" in msg:
            with open("live.log", "a") as f:
                f.write(timestamped + "\n")
        else:
            with open("research.log", "a") as f:
                f.write(timestamped + "\n")
    except Exception as e:
        print(f"Logging error: {e}", flush=True)

def git_commit(msg: str):
    try:
        if not (Path(__file__).parent / ".git").exists():
            subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "add", str(STRATEGY_FILE)], check=True, capture_output=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", msg], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', 'ignore').strip() if e.stderr else str(e)
        out_msg = e.stdout.decode('utf-8', 'ignore').strip() if e.stdout else ''
        log(f"Git commit failed: {err_msg} | {out_msg}")
    except Exception as e:
        log(f"Git commit failed: {e}")


# --- Scoring ---
def sharpe_sqrt_n(returns: list[float]) -> float:
    if len(returns) < 2:
        return -0.01 * max(1, 10 - len(returns))
    n = len(returns)
    mean_r = sum(returns) / n
    var = sum((r - mean_r)**2 for r in returns) / n
    std = math.sqrt(var) if var > 0 else 0.0
    
    # Structural Fix: Prevent standard deviation from approaching zero,
    # which causes the score to blow up to +/- infinity on small sample sizes.
    # For a binary outcome market, a standard deviation below 0.1 is usually 
    # just an artifact of tiny sample sizes (e.g. n=2 or n=3).
    std = max(0.1, std)
    
    return (mean_r / std) * math.sqrt(n)


# --- Results tracking ---
def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "generation\tvariant\tphase\tprediction\toutcome\tstatus\tprofit\t"
            "backtest_score\tlive_score\tdescription\n"
        )

def append_result(gen, var, phase, prediction, outcome, status, profit, bt_score, live_score, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{gen}\t{var}\t{phase}\t{prediction}\t{outcome}\t{status}\t"
                f"{profit:.4f}\t{bt_score:.4f}\t{live_score:.4f}\t{desc}\n")

def read_results_history(max_lines=80) -> str:
    if RESULTS_FILE.exists():
        lines = RESULTS_FILE.read_text().strip().split("\n")
        if len(lines) <= max_lines + 1:
            return "\n".join(lines)
        return f"{lines[0]}\n... ({len(lines)-max_lines-1} older rows truncated) ...\n" + "\n".join(lines[-max_lines:])
    return "(no results yet)"


# --- Data loading ---
def load_data():
    sys.path.insert(0, str(Path(__file__).parent))
    from train_data import load_raw_dataset
    return load_raw_dataset()


# --- Backtest engine ---
def run_backtest(feature_code: str, df: pd.DataFrame) -> dict:
    """
    Execute feature engineering + train + backtest.
    Returns dict with score, accuracy, n_trades, model, calibrator, features, returns.
    """
    result = {"score": -1.0, "accuracy": 0.0, "n_trades": 0, "returns": [],
              "model": None, "calibrator": None, "features": [], "error": None}
    try:
        # Execute the LLM-generated code to get functions
        namespace = {"__builtins__": __builtins__, "np": np, "pd": pd, "math": math}
        exec(feature_code, namespace)
        engineer_fn = namespace.get("engineer_features")
        hyperparam_fn = namespace.get("get_hyperparams")
        strategy_fn = namespace.get("strategy_filter")  # optional mutable strategy
        if not engineer_fn or not hyperparam_fn:
            result["error"] = "Missing engineer_features() or get_hyperparams()"
            return result

        # Feature engineering
        X = engineer_fn(df.copy())
        y = df['actual_outcome'].map({'Up': 1, 'Down': 0})

        # FIX 3: Random split by market to prevent LLM from overfitting a fixed test set
        unique_markets = list(df['market_slug'].unique())
        
        # We use random.Random() to avoid setting global numpy seed which might affect other things
        import random
        rng = random.Random()
        # Seed by feature_code hash or just totally random? Totally random to prevent overfitting
        rng.shuffle(unique_markets)
        
        n = len(unique_markets)
        train_mkts = unique_markets[:int(n*0.6)]
        val_mkts = unique_markets[int(n*0.6):int(n*0.8)]
        test_mkts = unique_markets[int(n*0.8):]

        train_idx = df['market_slug'].isin(train_mkts)
        val_idx = df['market_slug'].isin(val_mkts)
        test_idx = df['market_slug'].isin(test_mkts)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(X_train) < 10 or len(X_test) < 5:
            result["error"] = f"Not enough data: train={len(X_train)}, test={len(X_test)}"
            return result

        # Train
        params = hyperparam_fn()
        clf = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 1000),
            max_depth=params.get('max_depth', 2),
            learning_rate=params.get('learning_rate', 0.01),
            subsample=params.get('subsample', 0.5),
            colsample_bytree=params.get('colsample_bytree', 0.5),
            reg_alpha=params.get('reg_alpha', 1.0),
            reg_lambda=params.get('reg_lambda', 5.0),
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='logloss'
        )
        # Mispricing-aware sample weighting: upweight samples where market was wrong
        # This biases the model toward learning patterns that precede market errors
        train_cost_up = df[train_idx]['pm_cost_up'].values
        market_lean = train_cost_up - 0.5  # positive = market leans Up
        actual_up = y_train.values  # 1 = Up, 0 = Down
        # Positive when market leaned wrong way
        mispricing = np.where(actual_up == 1, np.maximum(-market_lean, 0),
                              np.maximum(market_lean, 0))
        sample_weights = 1.0 + mispricing * 10.0

        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
                sample_weight=sample_weights)

        # Calibrate
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(clf.predict_proba(X_val)[:, 1], y_val)

        # Backtest on holdout
        y_prob = calibrator.predict(clf.predict_proba(X_test)[:, 1])
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)

        # Simulate trading on holdout using strategy_filter (or default EV logic)
        trade_returns = []

        df_test = df[test_idx]
        costs_up = df_test['pm_cost_up'].values
        costs_down = df_test['pm_cost_down'].values
        secs_rem = df_test['seconds_remaining'].values

        for i in range(len(y_prob)):
            prob_up = y_prob[i]
            prob_down = 1.0 - prob_up
            cu, cd = costs_up[i], costs_down[i]

            if strategy_fn:
                try:
                    side, bet_frac = strategy_fn(
                        prob_up, prob_down, cu, cd,
                        X_test.iloc[i].to_dict(), secs_rem[i])
                except Exception:
                    side, bet_frac = None, 0.0
            else:
                # Default EV-threshold logic
                ev_up = prob_up - cu
                ev_down = prob_down - cd
                if max(ev_up, ev_down) > EV_MARGIN:
                    side = 'Up' if ev_up > ev_down else 'Down'
                    bet_frac = 1.0
                else:
                    side, bet_frac = None, 0.0

            if side and bet_frac > 0:
                cost = cu if side == 'Up' else cd
                actual = y_test.iloc[i]
                actual_match = (actual == 1) if side == 'Up' else (actual == 0)
                if actual_match:
                    trade_returns.append((1.0 - cost) * bet_frac)
                else:
                    trade_returns.append(-cost * bet_frac)

        score = sharpe_sqrt_n(trade_returns) if trade_returns else -1.0

        result.update({
            "score": score, "accuracy": acc, "n_trades": len(trade_returns),
            "returns": trade_returns, "model": clf, "calibrator": calibrator,
            "features": list(X.columns)
        })
    except Exception as e:
        result["error"] = str(e)
    return result


# --- LLM Agent ---
def create_client():
    return genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=60))

def build_population_summary(population: dict) -> str:
    """Build a summary of all active strategies for the LLM to avoid duplicates."""
    if not population:
        return "(no active strategies)"
    lines = []
    sorted_pop = sorted(population.items(), key=lambda x: x[1].get_score(), reverse=True)
    for idx, var in sorted_pop:
        phase = "LIVE" if var.promoted else "BACKTEST-ONLY" if var.backtest_score > 0 else "PENDING"
        score = sharpe_sqrt_n(var.live_returns) if var.live_returns else var.backtest_score
        lines.append(f"  Var {idx} [{phase}] Score={score:.3f} | BT={var.backtest_score:.3f} | "
                     f"Live PnL=${var.live_pnl:.2f} | Desc: {var.desc[:120]}")
    return "\n".join(lines)


def generate_single_variant(client, current_code, results_history, population_summary, temperature=0.9):
    """Generate exactly 1 focused mutation. Includes population context to avoid duplicates."""
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    prompt = textwrap.dedent(f"""\
    {program}

    ---

    ## Current Best Feature Engineering Code:
    ```python
    {current_code}
    ```

    ## Currently Active Strategies (DO NOT duplicate these approaches):
    {population_summary}

    ## Results History:
    ```
    {results_history}
    ```

    ---

    Generate exactly ONE new mutation. You may change ANY of the three functions:
    `engineer_features()`, `get_hyperparams()`, AND/OR `strategy_filter()`.
    
    **strategy_filter() is the HIGHEST-IMPACT mutation target** — it controls WHEN 
    to bet, WHICH side, and HOW MUCH. The features alone cannot create alpha in a 
    near-random 50/50 market, but smart conditional betting CAN.
    
    IMPORTANT RULES:
    - It MUST be meaningfully different from all currently active strategies listed above.
    - Do NOT re-try an approach that is already being tested.
    - Learn from the results history — avoid approaches that scored poorly.
    - Focus your full attention on crafting ONE high-quality, novel idea.
    - Think carefully about WHY your change should improve the score.
    - The code MUST include all three functions: engineer_features, get_hyperparams, strategy_filter.

    Output:
    1. A brief one-line description of what's different and why
    2. The complete code with `engineer_features()`, `get_hyperparams()`, AND `strategy_filter()` in a ```python block

    **Variant 1:**
    """)

    for attempt in range(10):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=8192)
            )
            if not response or not response.text:
                return []
            return parse_variants(response.text, 1)
        except Exception as e:
            if "500" in str(e):
                log(f"LLM 500 error (attempt {attempt+1}/10)")
                time.sleep(5)
                continue
            log(f"LLM error: {e}")
            return []
    return []

def parse_variants(text, expected):
    variants = []
    parts = re.split(r'\*\*Variant\s+\d+[:\s]*\*\*', text)
    if len(parts) > 1:
        for part in parts[1:]:
            desc = ""
            for line in part.strip().split("\n"):
                line = line.strip().strip("*").strip("-").strip()
                if line and not line.startswith("```"):
                    desc = line[:300]
                    break
            code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def engineer_features" in code and "def get_hyperparams" in code:
                    variants.append((code, desc or "unnamed"))
    if not variants:
        blocks = re.split(r'```python', text)
        for i in range(1, len(blocks)):
            desc = f"variant {len(variants)+1}"
            code_match = re.search(r'^\s*\n(.*?)```', blocks[i], re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def engineer_features" in code:
                    variants.append((code, desc))
    return variants[:expected]


# --- Model Variant ---
class ModelVariant:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc
        self.backtest_score = -1.0
        self.backtest_acc = 0.0
        self.backtest_n = 0
        self.live_returns = []
        self.live_pnl = 0.0
        self.live_wins = 0
        self.live_total = 0
        self.model = None
        self.calibrator = None
        self.features = []
        self.age = 0
        self.rounds_underwater = 0
        self.abstain_rounds = 0  # consecutive rounds with no trade
        self.promoted = False
        self.backtest_run = False

    def get_score(self):
        if self.live_returns:
            return sharpe_sqrt_n(self.live_returns)
        # Strategies with 0 live trades should NEVER outrank live-validated ones.
        # Use a negative score so they sit below any live strategy with score >= 0.
        if self.backtest_score > 0:
            return -0.001 - (1.0 / (self.backtest_score + 1))  # Better BT = closer to -0.001
        return -1.0 - 0.01 * self.age


def resolve_market_robustly(market, max_attempts=12) -> str | None:
    """
    Wait for the oracle to settle and return the correct outcome ('Up' or 'Down').
    Uses the CLOB midpoint prices to detect official contract settlement (0.995/0.005).
    """
    from main import PolymarketBTC5m
    slug = market.slug
    log(f"  Awaiting robust resolution for {slug}...")
    
    # Wait initially for 12 seconds
    time.sleep(12)
    
    for attempt in range(max_attempts):
        try:
            # Create a clean market instance for the expired ts
            temp_market = PolymarketBTC5m(initial_ts=market.current_ts)
            state = temp_market.get_prices()
            
            prices = state.get("prices") or state.get("outcome", {})
            up_mid = prices.get("Up", {}).get("midpoint", 0.0)
            down_mid = prices.get("Down", {}).get("midpoint", 0.0)
            
            log(f"    Resolution attempt {attempt+1}: Up mid = {up_mid:.3f}, Down mid = {down_mid:.3f}")
            
            if up_mid > 0.9 and down_mid < 0.1:
                return "Up"
            elif down_mid > 0.9 and up_mid < 0.1:
                return "Down"
                
        except Exception as e:
            log(f"    Error in robust resolution attempt {attempt+1}: {e}")
            
        time.sleep(5)
        
    # Fallback if oracle is extremely slow or something is wrong
    log(f"  ⚠️ Robust resolution timed out for {slug}. Falling back to midpoint comparison.")
    try:
        temp_market = PolymarketBTC5m(initial_ts=market.current_ts)
        state = temp_market.get_prices()
        prices = state.get("prices") or state.get("outcome", {})
        up_mid = prices.get("Up", {}).get("midpoint", 0.0)
        down_mid = prices.get("Down", {}).get("midpoint", 0.0)
        if up_mid > down_mid:
            return "Up"
        elif down_mid > up_mid:
            return "Down"
    except Exception:
        pass
    return None


# --- Live testing ---
def live_trading_loop(population: dict, population_lock: threading.Lock, nonlocal_state: dict = None):
    """Asynchronous thread that continually tests promoted models at optimal market times."""
    from main import PolymarketBTC5m
    from data_sources import get_btc_features
    from train_data import parse_features
    from paper_trade import position_size_detailed

    log("  [LIVE] Live trading thread started.")

    while True:
        try:
            with population_lock:
                promoted = {k: v for k, v in population.items() if v.promoted and v.model is not None}
                
                # Determine current #1 BEFORE trades happen to avoid look-ahead bias
                live_strategies = [(idx, var) for idx, var in population.items()
                                   if var.live_returns and var.promoted]
                current_best_idx_before = None
                if live_strategies:
                    current_best_idx_before, _ = max(live_strategies, key=lambda x: sharpe_sqrt_n(x[1].live_returns))
            
            if not promoted:
                time.sleep(10)
                continue

            market = PolymarketBTC5m()
            secs_remaining = market.market_end_time - time.time()

            # Optimal entry time is around 150 seconds remaining.
            # This gives 2.5 minutes of price action to the model, and 2.5 minutes left to expiry.
            if secs_remaining > 150:
                sleep_time = secs_remaining - 150
                time.sleep(sleep_time)
                secs_remaining = market.market_end_time - time.time()
            
            # If we somehow missed the window and it's too close to expiry, skip
            if secs_remaining < 60:
                time.sleep(5)
                continue

            log(f"  [LIVE] Entering {market.slug} with {secs_remaining:.0f}s remaining. Testing {len(promoted)} models.")

            # --- Build live DataFrame matching backtest data format ---
            # Read all prior snapshots of this market from the DB that
            # historical_collector is already writing to every ~20s.
            # This gives engineer_features() the same data depth and
            # temporal feature quality as during backtest training.
            import sqlite3 as _sqlite3
            db_path = Path(__file__).parent / "backtest_data.db"
            live_rows = []
            try:
                _conn = _sqlite3.connect(str(db_path))
                db_snaps = pd.read_sql_query(
                    "SELECT features, prices, seconds_remaining, timestamp "
                    "FROM market_snapshots WHERE market_slug = ? ORDER BY timestamp",
                    _conn, params=(market.slug,)
                )
                _conn.close()

                for _, snap in db_snaps.iterrows():
                    p = parse_features(snap['features'])
                    if p:
                        prices_data = json.loads(snap['prices']) if snap['prices'] else {}
                        p['market_slug'] = market.slug
                        p['timestamp'] = snap['timestamp']
                        p['seconds_remaining'] = snap['seconds_remaining']
                        p['actual_outcome'] = 'Up'  # placeholder, never used as feature
                        p['pm_cost_up'] = prices_data.get('Up', {}).get('best_ask', 0.5)
                        p['pm_cost_down'] = prices_data.get('Down', {}).get('best_ask', 0.5)
                        live_rows.append(p)
            except Exception as e:
                log(f"  [LIVE] DB query warning: {e}")

            # Fetch one fresh live snapshot for the most current data
            raw = get_btc_features()
            latest_parsed = parse_features(raw)
            secs_remaining = max(0, market.market_end_time - time.time())

            fill_up = position_size_detailed(BET_SIZE, outcome="Up", market=market)
            fill_down = position_size_detailed(BET_SIZE, outcome="Down", market=market)
            cost_up = fill_up["avg_fill_price"] if fill_up["shares"] > 0 else 1.0
            cost_down = fill_down["avg_fill_price"] if fill_down["shares"] > 0 else 1.0

            if latest_parsed:
                latest_parsed['market_slug'] = market.slug
                latest_parsed['timestamp'] = time.time()
                latest_parsed['seconds_remaining'] = secs_remaining
                latest_parsed['actual_outcome'] = 'Up'  # placeholder
                latest_parsed['pm_cost_up'] = cost_up
                latest_parsed['pm_cost_down'] = cost_down
                live_rows.append(latest_parsed)

            if len(live_rows) < 2:
                log("  [LIVE] Not enough history in DB for this market, skipping")
                time.sleep(10)
                continue

            df_live = pd.DataFrame(live_rows)
            
            # FIX 1: Add strike price features matching train_data logic
            if 'binance_price' in df_live.columns:
                df_live['strike_price'] = df_live['binance_price'].iloc[0]
                df_live['price_vs_strike'] = df_live['binance_price'] - df_live['strike_price']
                # Safe division to avoid ZeroDivisionError if strike_price is 0
                df_live['price_vs_strike_pct'] = (df_live['price_vs_strike'] / df_live['strike_price']).fillna(0) if df_live['strike_price'].iloc[0] > 0 else 0.0
            else:
                df_live['strike_price'] = 0.0
                df_live['price_vs_strike'] = 0.0
                df_live['price_vs_strike_pct'] = 0.0

            predictions = {}
            for idx, var in promoted.items():
                try:
                    namespace = {"__builtins__": __builtins__, "np": np, "pd": pd, "math": math}
                    exec(var.code, namespace)
                    engineer_fn = namespace.get("engineer_features")
                    if not engineer_fn:
                        continue

                    X_live_full = engineer_fn(df_live.copy())
                    X_live = X_live_full.iloc[[-1]].copy()
                    
                    for col in var.features:
                        if col not in X_live.columns:
                            X_live[col] = 0.0
                    X_live = X_live[var.features]

                    prob_up = var.calibrator.predict(var.model.predict_proba(X_live)[:, 1])[0]
                    prob_down = 1.0 - prob_up

                    live_strategy_fn = namespace.get("strategy_filter")
                    if live_strategy_fn:
                        try:
                            features_row = X_live.iloc[0].to_dict()
                            pred_side, bet_frac = live_strategy_fn(
                                prob_up, prob_down, cost_up, cost_down,
                                features_row, secs_remaining)
                            pred = pred_side if bet_frac > 0 else None
                        except Exception as e:
                            log(f"    [LIVE] Var {idx} strategy_filter error: {e}")
                            pred, bet_frac = None, 0.0
                    else:
                        bet_frac = 1.0
                        pred = None

                    if pred is None and bet_frac > 0:
                        ev_up = prob_up - cost_up
                        ev_down = prob_down - cost_down
                        if ev_up > ev_down and ev_up > EV_MARGIN:
                            pred = "Up"
                        elif ev_down > ev_up and ev_down > EV_MARGIN:
                            pred = "Down"

                    ev_up = prob_up - cost_up
                    ev_down = prob_down - cost_down
                    fill = fill_up if pred == "Up" else fill_down if pred == "Down" else None
                    thin = fill and fill["book_depth_usd"] < MIN_BOOK_DEPTH_USD
                    if thin:
                        pred = None

                    predictions[idx] = {
                        "prediction": pred, "prob_up": prob_up,
                        "ev_up": ev_up, "ev_down": ev_down,
                        "fill": fill, "thin": thin,
                        "invested": (BET_SIZE - fill["unfilled"]) * bet_frac if fill and pred else 0,
                        "shares": fill["shares"] * bet_frac if fill and pred else 0
                    }
                    log(f"    [LIVE] Var {idx}: P(Up)={prob_up:.3f} EV_Up={ev_up:+.3f} EV_Down={ev_down:+.3f} -> {pred or 'ABSTAIN'}")
                except Exception as e:
                    log(f"    [LIVE] Var {idx} inference error: {e}")
                    predictions[idx] = {"prediction": None, "invested": 0, "shares": 0, "thin": False}

            any_active_trade = any(p["prediction"] is not None for p in predictions.values())

            # Track abstentions per-strategy (even when no overall trade happens)
            with population_lock:
                for idx, p in predictions.items():
                    if idx not in population:
                        continue
                    var = population[idx]
                    if p["prediction"] is None:
                        var.abstain_rounds += 1
                    else:
                        var.abstain_rounds = 0

            if not any_active_trade:
                # Still increment age for all strategies even when skipping resolution
                with population_lock:
                    for idx in predictions:
                        if idx in population:
                            population[idx].age += 1
                log("  [LIVE] No active trades entered. Skipping resolution wait.")
                time.sleep(10)
                continue

            log("  [LIVE] Awaiting resolution...")
            while not market.is_expired:
                rem = market.market_end_time - time.time()
                if rem > 0:
                    time.sleep(min(15, rem + 1))

            actual = resolve_market_robustly(market)
            if not actual:
                log("  [LIVE] Could not determine outcome.")
                continue

            log(f"  [LIVE] Market resolved: {actual}")

            with population_lock:
                for idx, p in predictions.items():
                    if idx not in population:
                        continue # Culled by main thread during wait
                    var = population[idx]
                    var.age += 1
                    pred = p["prediction"]

                    if not pred:
                        continue

                    var.live_total += 1
                    correct = (pred == actual)
                    if correct:
                        var.live_wins += 1

                    if p["invested"] > 0:
                        revenue = p["shares"] if correct else 0.0
                        profit = revenue - p["invested"]
                        var.live_pnl += profit
                        var.live_returns.append(profit)
                    else:
                        var.live_returns.append(0.0)

                    if var.live_pnl <= 0:
                        var.rounds_underwater += 1
                    else:
                        var.rounds_underwater = 0

                    emoji = "✅" if correct else "❌"
                    live_score = sharpe_sqrt_n(var.live_returns)
                    log(f"    [LIVE] {emoji} Var {idx}: {pred} | Live PnL: ${var.live_pnl:.2f} | Score: {live_score:.3f}")

                # --- Cumulative #1 PnL tracking ---
                # Track PnL delta of whichever strategy WAS #1 before this trade
                if current_best_idx_before is not None and current_best_idx_before in population:
                    if current_best_idx_before in predictions and predictions[current_best_idx_before]["prediction"] is not None:
                        var = population[current_best_idx_before]
                        last_return = var.live_returns[-1] if var.live_returns else 0.0
                        nonlocal_state['cumulative_top1_pnl'] += last_return
                        nonlocal_state['cumulative_top1_trades'] += 1

                live_strategies = [(idx, var) for idx, var in population.items()
                                   if var.live_returns and var.promoted]
                if live_strategies:
                    current_best_idx, current_best_var = max(live_strategies,
                                                            key=lambda x: sharpe_sqrt_n(x[1].live_returns))
                    if current_best_idx != nonlocal_state.get('prev_top1_idx'):
                        if nonlocal_state.get('prev_top1_idx') is not None:
                            nonlocal_state['top1_rotations'] += 1
                        nonlocal_state['prev_top1_idx'] = current_best_idx
                        log(f"    📊 #1 rotated to Var {current_best_idx}")

                # --- Best LIVE strategy ratchet ---
                # After recording results, find the best live-validated strategy
                # (must have at least 1 actual trade in live_returns)
                best_live_candidate = None
                best_live_candidate_score = -999.0
                for idx, var in population.items():
                    if var.live_returns and var.promoted:
                        ls = sharpe_sqrt_n(var.live_returns)
                        if ls > best_live_candidate_score:
                            best_live_candidate_score = ls
                            best_live_candidate = (idx, var)

                if best_live_candidate:
                    cidx, cvar = best_live_candidate
                    is_new_leader = cidx != nonlocal_state.get('best_live_idx')
                    is_score_improvement = best_live_candidate_score > nonlocal_state.get('best_live_score', -999.0)
                    
                    nonlocal_state['best_live_score'] = best_live_candidate_score
                    nonlocal_state['best_live_code'] = cvar.code
                    nonlocal_state['best_live_desc'] = cvar.desc
                    nonlocal_state['best_live_idx'] = cidx

                    if is_new_leader or is_score_improvement:
                        # Save as best_strategy.py — this is the LIVE-proven best
                        STRATEGY_FILE.write_text(cvar.code)
                        git_commit(f"LIVE Best (Var {cidx}): {cvar.desc[:60]} [Live Score: {best_live_candidate_score:.3f}]")
                        if is_new_leader:
                            log(f"    ⭐ New best LIVE strategy: Var {cidx} | Score: {best_live_candidate_score:.3f} | PnL: ${cvar.live_pnl:.2f}")

                # --- Save all live strategy code to strategies/ folder ---
                for idx, var in population.items():
                    if var.promoted:
                        strat_path = STRATEGIES_DIR / f"var_{idx}.py"
                        header = f"# Var {idx} | BT: {var.backtest_score:.3f} | Live PnL: ${var.live_pnl:.2f} | Trades: {var.live_total}\n"
                        header += f"# Desc: {var.desc[:200]}\n"
                        # Persist all live stats as JSON so they survive restarts
                        live_stats = {
                            'live_returns': [float(x) for x in var.live_returns],
                            'live_pnl': float(var.live_pnl),
                            'live_wins': int(var.live_wins),
                            'live_total': int(var.live_total),
                            'age': int(var.age),
                            'rounds_underwater': int(var.rounds_underwater),
                            'abstain_rounds': int(var.abstain_rounds),
                            'backtest_score': float(var.backtest_score),
                            'backtest_acc': float(var.backtest_acc),
                            'backtest_n': int(var.backtest_n),
                        }
                        header += f"# Stats: {json.dumps(live_stats)}\n\n"
                        strat_path.write_text(header + var.code)

        except Exception as e:
            log(f"  [LIVE] Error in live loop: {e}")
            traceback.print_exc()
            time.sleep(5)


# --- Default feature code ---
DEFAULT_CODE = '''
def engineer_features(df):
    """Baseline feature engineering with temporal context."""
    X = df.drop(columns=['market_slug', 'timestamp', 'actual_outcome'])
    trend_cols = ['mean_obi', 'mean_delta', 'mean_momentum', 'binance_price', 'mean_funding', 'price_std']
    for col in trend_cols:
        if col in X.columns:
            X[f'{col}_diff_3'] = df.groupby('market_slug')[col].diff(3).fillna(0)
            X[f'{col}_diff_6'] = df.groupby('market_slug')[col].diff(6).fillna(0)
            X[f'{col}_roll_mean_6'] = df.groupby('market_slug')[col].transform(
                lambda x: x.rolling(6, min_periods=1).mean())
            X[f'{col}_roc_6'] = df.groupby('market_slug')[col].pct_change(
                periods=6).fillna(0).replace([float('inf'), float('-inf')], 0)
    return X

def get_hyperparams():
    return {
        'n_estimators': 1000, 'max_depth': 2, 'learning_rate': 0.01,
        'subsample': 0.5, 'colsample_bytree': 0.5,
        'reg_alpha': 1.0, 'reg_lambda': 5.0,
    }

def strategy_filter(prob_up, prob_down, cost_up, cost_down, features_row, seconds_remaining):
    """Default EV-threshold strategy."""
    ev_up = prob_up - cost_up
    ev_down = prob_down - cost_down
    best_ev = max(ev_up, ev_down)
    if best_ev > 0.03:
        if ev_up > ev_down:
            return "Up", 1.0
        else:
            return "Down", 1.0
    return None, 0.0
'''


# --- Main loop ---
def main():
    log("╔══════════════════════════════════════════════════╗")
    log("║  ML Autoresearch Loop — Sharpe*sqrt(N) Scoring  ║")
    log(f"║  Model: {MODEL}                    ║")
    log("╚══════════════════════════════════════════════════╝")

    if not API_KEY:
        log("ERROR: No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        sys.exit(1)

    client = create_client()
    init_results()

    log("Loading historical data...")
    df = load_data()
    log(f"Loaded {len(df)} samples across {df['market_slug'].nunique()} markets")

    # --- Load saved strategies from disk (preserves progress across restarts) ---
    def _strip_header(code_text: str) -> str:
        """Strip comment headers we add to saved strategy files."""
        lines = code_text.split('\n')
        clean = [l for l in lines if not l.startswith('# Var ') and not l.startswith('# Desc:') and not l.startswith('# Stats:')]
        return '\n'.join(clean).strip()

    def _parse_live_stats(raw_text: str) -> dict | None:
        """Extract persisted live stats JSON from strategy file header."""
        for line in raw_text.split('\n'):
            if line.startswith('# Stats:'):
                try:
                    return json.loads(line[8:].strip())
                except (json.JSONDecodeError, ValueError):
                    return None
        return None

    def _is_valid_strategy(code: str) -> bool:
        return 'def engineer_features' in code and 'def get_hyperparams' in code

    # 1. Load best_strategy.py as the seed / mutation parent
    initial_code = DEFAULT_CODE
    initial_desc = "Baseline temporal features"
    initial_bt_score = -999.0
    if STRATEGY_FILE.exists():
        saved_code = _strip_header(STRATEGY_FILE.read_text())
        if _is_valid_strategy(saved_code):
            initial_code = saved_code
            initial_desc = "Loaded from best_strategy.py"
            initial_bt_score = 0.0  # Will be properly scored during backtest phase
            log(f"  ✅ Loaded existing best_strategy.py as initial seed")
        else:
            log(f"  ⚠️ best_strategy.py exists but missing required functions, using default")

    population = {0: ModelVariant(initial_code, initial_desc)}

    # 2. Load all strategies from strategies/ folder
    loaded_from_disk = 0
    loaded_with_stats = 0
    if STRATEGIES_DIR.exists():
        for strat_file in sorted(STRATEGIES_DIR.glob("var_*.py")):
            try:
                # Extract var ID from filename (var_7.py -> 7)
                var_id = int(strat_file.stem.split('_')[1])
                if var_id == 0:
                    continue  # Already loaded as seed above
                raw = strat_file.read_text()
                code = _strip_header(raw)
                if not _is_valid_strategy(code):
                    continue
                # Extract description from header comment if present
                desc = f"Loaded from {strat_file.name}"
                for line in raw.split('\n'):
                    if line.startswith('# Desc:'):
                        desc = line[7:].strip()
                        break
                variant = ModelVariant(code, desc)
                
                # Restore persisted live stats if available
                stats = _parse_live_stats(raw)
                if stats:
                    variant.live_returns = stats.get('live_returns', [])
                    variant.live_pnl = stats.get('live_pnl', 0.0)
                    variant.live_wins = stats.get('live_wins', 0)
                    variant.live_total = stats.get('live_total', 0)
                    variant.age = stats.get('age', 0)
                    variant.rounds_underwater = stats.get('rounds_underwater', 0)
                    variant.abstain_rounds = stats.get('abstain_rounds', 0)
                    variant.backtest_score = stats.get('backtest_score', -1.0)
                    variant.backtest_acc = stats.get('backtest_acc', 0.0)
                    variant.backtest_n = stats.get('backtest_n', 0)
                    loaded_with_stats += 1
                
                population[var_id] = variant
                loaded_from_disk += 1
            except (ValueError, IndexError):
                continue  # Skip files with non-numeric IDs
    if loaded_from_disk > 0:
        log(f"  ✅ Loaded {loaded_from_disk} strategies from strategies/ folder ({loaded_with_stats} with persisted live stats)")

    generation = 1

    # Persistent best-ever tracking — survives culling so we never lose the best mutation parent
    best_ever_code = initial_code
    best_ever_score = initial_bt_score
    best_ever_desc = initial_desc

    # Separate tracking for best LIVE-validated strategy (must have actual trades)
    best_live_score = -999.0
    best_live_code = None
    best_live_desc = None
    best_live_idx = None

    # Restore best-ever and best-live from persisted stats in loaded population
    active_population = {}
    culled_count = 0
    for idx, var in population.items():
        # Apply culling filters on reload to avoid reactivating previously culled strategies
        is_culled = False
        if var.rounds_underwater >= MAX_UNDERWATER_ROUNDS:
            is_culled = True
        elif var.live_total >= MAX_UNPROFITABLE_TRADES and var.live_pnl <= 0:
            is_culled = True
        elif var.abstain_rounds >= MAX_ABSTAIN_ROUNDS:
            is_culled = True

        if is_culled:
            culled_count += 1
            continue

        active_population[idx] = var

        # Restore best_ever from backtest scores
        if var.backtest_score > best_ever_score:
            best_ever_score = var.backtest_score
            best_ever_code = var.code
            best_ever_desc = var.desc
        # Restore best_live from persisted live returns
        if var.live_returns:
            ls = sharpe_sqrt_n(var.live_returns)
            if ls > best_live_score:
                best_live_score = ls
                best_live_code = var.code
                best_live_desc = var.desc
                best_live_idx = idx
            # Mark as promoted + backtest_run since it had live trades
            var.promoted = True
            var.backtest_run = True
            # Reload the model from disk if available
            model_path = MODELS_DIR / f"var_{idx}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        saved = pickle.load(f)
                    var.model = saved["model"]
                    var.calibrator = saved["calibrator"]
                    var.features = saved["features"]
                except Exception as e:
                    log(f"  ⚠️ Failed to reload model for Var {idx}: {e}")

    population = active_population
    if culled_count > 0:
        log(f"  ☠️ Filtered out {culled_count} previously culled strategies from memory. Active: {len(population)}")


    if best_live_idx is not None:
        log(f"  ✅ Restored best live strategy: Var {best_live_idx} | Score: {best_live_score:.3f} | PnL: ${population[best_live_idx].live_pnl:.2f}")
    if best_ever_score > -999:
        log(f"  ✅ Restored best-ever BT score: {best_ever_score:.3f} ({best_ever_desc[:60]})")

    # Shared state for live thread to update best-live tracking (mutable dict for closure)
    nonlocal_state = {
        'best_live_score': best_live_score,
        'best_live_code': best_live_code,
        'best_live_desc': best_live_desc,
        'best_live_idx': best_live_idx,
        'cumulative_top1_pnl': 0.0,     # PnL from following the #1 strategy
        'cumulative_top1_trades': 0,     # total trades while following #1
        'top1_rotations': 0,             # how many times #1 has changed
        'prev_top1_idx': None,           # previous #1 strategy index
    }

    # --- Startup retrain: refresh all live strategies before going live ---
    startup_retrain = [(idx, var) for idx, var in population.items()
                       if var.promoted and var.live_returns]
    if startup_retrain:
        log(f"\n--- Startup Retraining ({len(startup_retrain)} live strategies) ---")
        for idx, var in startup_retrain:
            log(f"  Retraining Var {idx} ({var.desc[:50]})...")
            bt_runs = []
            best_bt = None
            for trial in range(3):
                bt = run_backtest(var.code, df)
                if bt["error"]:
                    continue
                bt_runs.append(bt)
                if best_bt is None or bt["score"] > best_bt["score"]:
                    best_bt = bt

            if not bt_runs:
                log(f"    ⚠️ All retrain attempts errored — keeping existing model")
                continue

            scores = sorted([b["score"] for b in bt_runs])
            median_score = scores[len(scores) // 2]
            old_score = var.backtest_score
            new_score = RETRAIN_EMA_ALPHA * median_score + (1 - RETRAIN_EMA_ALPHA) * old_score
            log(f"    Old BT: {old_score:.3f} | Fresh BT (3 runs): {[f'{s:.3f}' for s in scores]} median={median_score:.3f} | EMA: {new_score:.3f}")

            var.backtest_score = new_score
            if median_score > old_score:
                var.model = best_bt["model"]
                var.calibrator = best_bt["calibrator"]
                var.features = best_bt["features"]
                model_path = MODELS_DIR / f"var_{idx}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump({"model": best_bt["model"], "calibrator": best_bt["calibrator"],
                                 "features": best_bt["features"]}, f)
                log(f"    ✅ Model weights upgraded (median {median_score:.3f} > old {old_score:.3f})")
            else:
                log(f"    ↔️ Keeping existing weights ({median_score:.3f} ≤ {old_score:.3f})")
        log("--- Startup Retraining Complete ---\n")

    # Start the live trading loop in a separate thread
    population_lock = threading.Lock()
    live_thread = threading.Thread(
        target=live_trading_loop,
        args=(population, population_lock, nonlocal_state),
        daemon=True
    )
    live_thread.start()

    while True:
        try:
            with population_lock:
                pop_len = len(population)
            log(f"\n═══ Generation {generation} | Population: {pop_len}/{MAX_POPULATION} ═══")

            # 1. Generate 1 new mutation per round (focused LLM attention)
            if pop_len < MAX_POPULATION:
                # Prefer the best LIVE-validated strategy as mutation parent,
                # fall back to best backtest-only if no live strategy exists yet
                if nonlocal_state['best_live_code'] is not None:
                    parent_code = nonlocal_state['best_live_code']
                    parent_desc = nonlocal_state['best_live_desc']
                    parent_score = nonlocal_state['best_live_score']
                    parent_label = "LIVE"
                else:
                    parent_code = best_ever_code
                    parent_desc = best_ever_desc
                    parent_score = best_ever_score
                    parent_label = "BT"
                with population_lock:
                    pop_summary = build_population_summary(population)

                log(f"Generating 1 focused mutation (parent [{parent_label}]: {parent_desc[:60]}, score: {parent_score:.3f})...")
                temp = 0.9
                if best_ever_score > 0.5:
                    temp = 0.5
                variants = generate_single_variant(
                    client, parent_code, read_results_history(),
                    pop_summary, temp
                )

                if variants:
                    with population_lock:
                        next_id = max(population.keys()) + 1 if population else 1
                        code, desc = variants[0]
                        population[next_id] = ModelVariant(code, desc)
                    log(f"  + Var {next_id}: {desc[:80]}")
                else:
                    log("  No valid variant generated this round")

            # 2. Backtest all un-backtested variants (retry failures immediately)
            log("\n--- Backtest Phase ---")
            with population_lock:
                to_backtest = [(idx, var.code, var.desc) for idx, var in population.items() if not var.backtest_run]
                
            for idx, code, desc in to_backtest:
                log(f"  Backtesting Var {idx}: {desc[:60]}...")
                bt = run_backtest(code, df)
                retries = 0
                while bt["error"] and retries < 3:
                    retries += 1
                    log(f"    ❌ Error: {bt['error']}")
                    log(f"    Regenerating (retry {retries}/3)...")
                    with population_lock:
                        pop_summary = build_population_summary(population)
                    new_variants = generate_single_variant(
                        client, code, read_results_history(),
                        pop_summary, 0.9
                    )
                    if new_variants:
                        code, desc = new_variants[0]
                        log(f"    → New Var {idx}: {desc[:80]}")
                        bt = run_backtest(code, df)
                    else:
                        break
                        
                with population_lock:
                    if idx not in population:
                        continue # Was somehow culled (unlikely since it was pending backtest)
                    var = population[idx]
                    var.code = code
                    var.desc = desc
                    
                    if bt["error"]:
                        log(f"    ❌ Failed after {retries} retries: {bt['error']}")
                        var.backtest_score = -999
                    else:
                        var.backtest_score = bt["score"]
                        var.backtest_acc = bt["accuracy"]
                        var.backtest_n = bt["n_trades"]
                        log(f"    Score: {bt['score']:.3f} | Acc: {bt['accuracy']:.3f} | Trades: {bt['n_trades']}")

                        # Track best-ever code for mutation parent persistence
                        if bt["score"] > best_ever_score:
                            best_ever_score = bt["score"]
                            best_ever_code = var.code
                            best_ever_desc = var.desc
                            log(f"    ⭐ New best-ever score: {best_ever_score:.3f}")
                            
                            # Git Ratchet: save BT-best strategy ONLY if no live-validated best exists yet
                            if nonlocal_state['best_live_code'] is None:
                                STRATEGY_FILE.write_text(var.code)
                                git_commit(f"Gen {generation} (Var {idx}): {var.desc[:60]} [BT Score: {bt['score']:.3f}]")

                        if bt["score"] >= BACKTEST_SCORE_THRESHOLD:
                            var.model = bt["model"]
                            var.calibrator = bt["calibrator"]
                            var.features = bt["features"]
                            var.promoted = True
                            # Save model
                            model_path = MODELS_DIR / f"var_{idx}.pkl"
                            with open(model_path, "wb") as f:
                                pickle.dump({"model": bt["model"], "calibrator": bt["calibrator"],
                                             "features": bt["features"]}, f)
                            log(f"    🚀 PROMOTED to live! Saved to {model_path.name}")
                            # Save strategy code to strategies/ folder
                            strat_path = STRATEGIES_DIR / f"var_{idx}.py"
                            header = f"# Var {idx} | BT: {bt['score']:.3f} | Desc: {desc[:200]}\n\n"
                            strat_path.write_text(header + code)
                        else:
                            log(f"    Below threshold ({BACKTEST_SCORE_THRESHOLD}), not promoted")

                        var.backtest_run = True

                append_result(generation, idx, "backtest", "-", "-",
                              "pass" if bt.get("score", -1) >= BACKTEST_SCORE_THRESHOLD else "fail",
                              0.0, bt.get("score", -1), 0.0, desc[:100])

            # 3. Live testing is now asynchronous
            # run_live_round is handled by live_trading_loop thread

            # 3b. Periodic retraining of promoted models
            if generation % RETRAIN_INTERVAL == 0 and generation > 0:
                log("\n--- Retraining Phase ---")
                # Reload data so retraining uses the freshest snapshots
                df = load_data()
                log(f"  Reloaded data: {len(df)} samples across {df['market_slug'].nunique()} markets")

                with population_lock:
                    retrain_candidates = [(idx, var) for idx, var in population.items()
                                          if var.promoted and var.model is not None and var.live_returns]

                for idx, var in retrain_candidates:
                    log(f"  Retraining Var {idx} ({var.desc[:50]})...")

                    # Run 3 independent backtests (each gets a different random split)
                    # and take the median to reduce split variance noise
                    bt_runs = []
                    best_bt = None
                    for trial in range(3):
                        bt = run_backtest(var.code, df)
                        if bt["error"]:
                            continue
                        bt_runs.append(bt)
                        if best_bt is None or bt["score"] > best_bt["score"]:
                            best_bt = bt

                    if not bt_runs:
                        log(f"    ⚠️ All retrain attempts errored — keeping existing model")
                        continue

                    scores = sorted([b["score"] for b in bt_runs])
                    median_score = scores[len(scores) // 2]

                    old_score = var.backtest_score
                    new_score = RETRAIN_EMA_ALPHA * median_score + (1 - RETRAIN_EMA_ALPHA) * old_score
                    log(f"    Old BT: {old_score:.3f} | Fresh BT (3 runs): {[f'{s:.3f}' for s in scores]} median={median_score:.3f} | EMA: {new_score:.3f}")

                    with population_lock:
                        if idx not in population:
                            continue
                        population[idx].backtest_score = new_score

                        # Only replace model weights if the median fresh score
                        # EXCEEDS the current score — never downgrade a working model
                        if median_score > old_score:
                            population[idx].model = best_bt["model"]
                            population[idx].calibrator = best_bt["calibrator"]
                            population[idx].features = best_bt["features"]
                            # Save updated model to disk
                            model_path = MODELS_DIR / f"var_{idx}.pkl"
                            with open(model_path, "wb") as f:
                                pickle.dump({"model": best_bt["model"], "calibrator": best_bt["calibrator"],
                                             "features": best_bt["features"]}, f)
                            log(f"    ✅ Model weights upgraded (median {median_score:.3f} > old {old_score:.3f})")
                        else:
                            log(f"    ↔️ Score stable/declining ({median_score:.3f} ≤ {old_score:.3f}), keeping existing weights")

            # 4. Culling
            log("\n--- Culling ---")
            with population_lock:
                to_kill = []
                for idx, var in population.items():
                    if var.backtest_score == -999:
                        log(f"  ☠️ Killed Var {idx} (backtest crash)")
                        to_kill.append(idx)
                    elif var.backtest_run and not var.promoted:
                        log(f"  ☠️ Killed Var {idx} (failed backtest)")
                        to_kill.append(idx)
                    elif var.rounds_underwater >= MAX_UNDERWATER_ROUNDS:
                        log(f"  ☠️ Killed Var {idx} (underwater {MAX_UNDERWATER_ROUNDS} rounds)")
                        to_kill.append(idx)
                    elif var.live_total >= MAX_UNPROFITABLE_TRADES and var.live_pnl <= 0:
                        log(f"  ☠️ Killed Var {idx} (unprofitable after {MAX_UNPROFITABLE_TRADES}+ trades)")
                        to_kill.append(idx)
                    elif var.promoted and var.abstain_rounds >= MAX_ABSTAIN_ROUNDS:
                        log(f"  ☠️ Killed Var {idx} (abstained {MAX_ABSTAIN_ROUNDS} consecutive rounds)")
                        to_kill.append(idx)

                if len(to_kill) == len(population) and population:
                    best = max(population.keys(), key=lambda k: population[k].get_score())
                    to_kill.remove(best)
                    log(f"  🛡️ Spared Var {best}")
                for idx in to_kill:
                    del population[idx]

            # 5. Leaderboard
            cum_pnl = nonlocal_state['cumulative_top1_pnl']
            cum_trades = nonlocal_state['cumulative_top1_trades']
            cum_rotations = nonlocal_state['top1_rotations']
            log(f"\n--- Leaderboard --- [#1 Cumulative PnL: ${cum_pnl:.2f} | Trades: {cum_trades} | Rotations: {cum_rotations}]")
            with population_lock:
                sorted_pop = sorted(population.items(), key=lambda x: x[1].get_score(), reverse=True)
                live_best_idx = nonlocal_state.get('best_live_idx')
                for i, (idx, var) in enumerate(sorted_pop):
                    phase = "LIVE" if var.promoted else "BACKTEST"
                    if var.live_returns:
                        score = sharpe_sqrt_n(var.live_returns)
                        trades_label = f"Trades: {len(var.live_returns)}"
                    else:
                        score = var.backtest_score
                        trades_label = f"Trades: 0 (BT only, abstain: {var.abstain_rounds})"
                    best_tag = " 🏆" if idx == live_best_idx else ""
                    wr = var.live_wins / var.live_total if var.live_total > 0 else 0
                    log(f"  #{i+1}: Var {idx} [{phase}] | Score: {score:.3f} | "
                        f"BT: {var.backtest_score:.3f} | Live PnL: ${var.live_pnl:.2f} | "
                        f"Wins: {var.live_wins}/{var.live_total} ({wr:.0%}) | {trades_label} | Age: {var.age}{best_tag}")

            # Sleep to yield if population is full and no backtests were run
            if not to_backtest and pop_len >= MAX_POPULATION:
                time.sleep(10)

            # Periodic data reload to benefit from accumulating snapshots
            if generation % 50 == 0:
                log("Reloading training data from disk...")
                df = load_data()
                log(f"  Now: {len(df)} samples across {df['market_slug'].nunique()} markets")

            generation += 1

        except KeyboardInterrupt:
            log("\nInterrupted. Exiting.")
            break
        except Exception as e:
            log(f"Generation {generation} failed: {e}")
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    main()
