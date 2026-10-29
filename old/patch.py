import re

with open('autoresearch.py', 'r') as f:
    content = f.read()

# 1. Change VARIANTS_PER_GENERATION
content = content.replace("VARIANTS_PER_GENERATION = 3", "VARIANTS_PER_GENERATION = 5")

# 2. Refactor run_generation
# Replace the run_generation and main loop
run_gen_start = content.find("def run_generation(")
new_run_gen = """
def run_tournament(client: genai.Client, generation: int, temperature: float = 0.9) -> bool:
    log(f"\\n═══ Tournament {generation} ═══")
    
    # --- Read current strategy ---
    current_strategy = STRATEGY_FILE.read_text() if STRATEGY_FILE.exists() else ""
    parent_hash = git_current_hash()
    log(f"Parent commit: {parent_hash}")
    
    # --- Generate variants via LLM ---
    log(f"Asking {MODEL} for {VARIANTS_PER_GENERATION} strategy variants (temp={temperature:.1f})...")
    results_history = read_results_history()
    
    # Get a sample market state just for context
    market_prices_sample = {}
    try:
        from main import PolymarketBTC5m
        temp_mkt = PolymarketBTC5m()
        state = temp_mkt.get_prices()
        prices = state.get("prices", {})
        for side in ("Up", "Down"):
            market_prices_sample[side] = prices.get(side, {})
        market_prices_sample["seconds_remaining"] = max(0, temp_mkt.market_end_time - time.time())
    except Exception:
        pass
        
    try:
        features = fetch_features()
    except Exception:
        features = []

    variants = generate_strategy_variants(
        client, current_strategy, results_history, features, market_prices_sample, temperature=temperature
    )

    if not variants:
        log("No valid variants generated. Using current strategy only.")
        variants = [(current_strategy, "current strategy (no mutation)")]

    log(f"Got {len(variants)} variants for the tournament")
    
    # Initialize tournament trackers
    # trackers[idx] = {"code": ..., "desc": ..., "pnl": 0.0, "wins": 0, "total": 0}
    trackers = {
        0: {"code": current_strategy, "desc": "current best (control)", "pnl": 0.0, "wins": 0, "total": 0}
    }
    for i, (code, desc) in enumerate(variants, 1):
        trackers[i] = {"code": code, "desc": desc, "pnl": 0.0, "wins": 0, "total": 0}

    # Run 3 rounds
    ROUNDS = 3
    for round_num in range(1, ROUNDS + 1):
        log(f"\\n--- Tournament {generation} - Round {round_num}/{ROUNDS} ---")
        
        # 1. Fetch live features
        log("Fetching BTC features from exchanges...")
        try:
            features = fetch_features()
        except Exception as e:
            log(f"Feature fetch failed: {e}. Using empty features.")
            features = []

        # 2. Get current market state
        log("Fetching Polymarket state...")
        try:
            market_state = get_market_state()
            market_ts = int(market_state.get("slug", "-0").split("-")[-1])
            log(f"Market: {market_state.get('slug', 'unknown')}, ends={market_state.get('end_time', '?')}")
        except Exception as e:
            log(f"Market fetch failed: {e}")
            market_state = {}
            market_ts = 0

        # 3. Run all variants concurrently
        import concurrent.futures
        predictions = []

        def worker(p_idx):
            code = trackers[p_idx]["code"]
            desc = trackers[p_idx]["desc"]
            
            from main import PolymarketBTC5m
            worker_mkt = PolymarketBTC5m()

            def _get_market_prices():
                try:
                    state = worker_mkt.get_prices()
                    prices = state.get("prices", {})
                    result = {}
                    for side in ("Up", "Down"):
                        result[side] = prices.get(side, {})
                    import time
                    result["seconds_remaining"] = max(0, worker_mkt.market_end_time - time.time())
                    return result
                except Exception:
                    return {}

            pred = run_strategy(code, get_market_prices=_get_market_prices)
            outcome_pred = pred[0] if pred else None
            bet_amount = pred[1] if pred else 0.0

            shares = 0.0; invested = 0.0; avg_fill_price = 0.0; book_depth = 0.0; spread = 0.0; thin_book = False
            
            if outcome_pred:
                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from paper_trade import position_size_detailed
                    from main import PolymarketBTC5m
                    live_mkt = PolymarketBTC5m()
                    fill = position_size_detailed(bet_amount, outcome=outcome_pred, market=live_mkt)
                    shares = fill["shares"]; invested = bet_amount - fill["unfilled"]
                    avg_fill_price = fill["avg_fill_price"]; book_depth = fill["book_depth_usd"]; spread = fill["spread"]
                    if book_depth < MIN_BOOK_DEPTH_USD: thin_book = True
                except Exception as e:
                    log(f"  Variant {p_idx}: immediate orderbook simulation failed: {e}")

            return {
                "idx": p_idx, "prediction": outcome_pred, "bet_amount": bet_amount,
                "shares": shares, "invested": invested, "avg_fill_price": avg_fill_price,
                "book_depth": book_depth, "spread": spread, "thin_book": thin_book
            }

        log("Running strategies concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(trackers)) as executor:
            futures = [executor.submit(worker, idx) for idx in trackers.keys()]
            for future in concurrent.futures.as_completed(futures):
                p = future.result()
                predictions.append(p)
                thin_tag = " [THIN]" if p.get("thin_book") else ""
                log(f"  Variant {p['idx']}: predicts {p['prediction']} bet=${p['bet_amount']:.2f} "
                    f"fill@${p['avg_fill_price']:.4f}{thin_tag}")

        predictions.sort(key=lambda x: x["idx"])

        # 4. Wait for market resolution
        if market_state.get("status") == "live" and market_ts > 0:
            import time
            from main import PolymarketBTC5m
            temp_market = PolymarketBTC5m(initial_ts=market_ts)
            secs_remaining = temp_market.market_end_time - time.time()
            if secs_remaining > 0:
                log(f"Waiting {secs_remaining:.0f}s for market to expire...")
                while not temp_market.is_expired:
                    rem = temp_market.market_end_time - time.time()
                    if rem > 0:
                        wait = min(10, rem + 1)
                        time.sleep(wait)
                        if rem > 10: log(f"  {rem:.0f}s remaining...")
            else:
                log("Market already expired, fetching resolution...")
            time.sleep(3)

        # 5. Determine outcome
        log("Resolving market outcome...")
        resolution_mids = {}
        try:
            from main import PolymarketBTC5m
            resolved_market = PolymarketBTC5m(initial_ts=market_ts) if market_ts > 0 else PolymarketBTC5m()
            resolved_state = resolved_market.get_prices()
            actual_outcome = determine_outcome(resolved_state)
            res_prices = resolved_state.get("outcome") or resolved_state.get("prices", {})
            for side in ("Up", "Down"):
                resolution_mids[side] = res_prices.get(side, {}).get("midpoint", 0.0)
        except Exception as e:
            log(f"Resolution error: {e}")
            actual_outcome = None

        if actual_outcome:
            log(f"Market resolved: {actual_outcome} won! (mids: Up={resolution_mids.get('Up',0):.4f}, Down={resolution_mids.get('Down',0):.4f})")
        else:
            actual_outcome = determine_outcome(market_state)
            if actual_outcome:
                log(f"Using pre-resolution data: {actual_outcome}")
            else:
                log("Cannot determine outcome. Skipping scoring for this round.")
                continue

        # 6. Score and track PnL
        for p in predictions:
            idx = p["idx"]
            outcome_pred = p["prediction"]
            res_mid = resolution_mids.get(outcome_pred, 0.0) if outcome_pred else 0.0
            status = "crash"; profit = 0.0; roi = 0.0
            
            if outcome_pred:
                trackers[idx]["total"] += 1
                correct = (outcome_pred == actual_outcome)
                status = "win" if correct else "loss"
                if correct: trackers[idx]["wins"] += 1
                
                if p["invested"] > 0:
                    revenue = p["shares"] * 1.00 if correct else 0.0
                    profit = revenue - p["invested"]
                    roi = profit / p["invested"]
                
                if p["thin_book"]:
                    profit = 0.0 # Reject thin book profits for fitness
                    
                trackers[idx]["pnl"] += profit
                
            append_result(
                gen=generation, var=idx, commit=parent_hash, prediction=outcome_pred or "CRASH",
                bet_amount=p["bet_amount"], outcome=actual_outcome, status=status, profit=profit, roi=roi,
                desc=trackers[idx]["desc"], avg_fill=p["avg_fill_price"], book_depth=p["book_depth"],
                spread=p["spread"], resolution_mid=res_mid
            )
            
            emoji = "✅" if status == "win" else "❌"
            log(f"  {emoji} Variant {idx}: {status} | Profit: ${profit:.2f} | Total PnL: ${trackers[idx]['pnl']:.2f}")

    # End of Tournament - Evaluate and Git Ratchet
    log(f"\\n═══ Tournament {generation} Results ═══")
    best_variant = None
    best_pnl = trackers[0]["pnl"] # Control PnL to beat
    
    for idx, data in trackers.items():
        win_rate = data["wins"] / data["total"] if data["total"] > 0 else 0
        log(f"Variant {idx}: PnL = ${data['pnl']:.2f} | Wins = {data['wins']}/{data['total']} ({win_rate:.1%}) | {data['desc'][:50]}")
        
        # Mutation must BEAT control PnL strictly, or if control is negative, it must be less negative
        if idx > 0 and data["pnl"] > best_pnl:
            best_pnl = data["pnl"]
            best_variant = idx
            
    mutation_adopted = False
    if best_variant is not None:
        log(f"\\nAdvancing to variant {best_variant} (PnL: ${best_pnl:.2f})")
        STRATEGY_FILE.write_text(trackers[best_variant]["code"])
        git_commit(f"tourney{generation}: {trackers[best_variant]['desc'][:60]} [WIN]")
        log(f"Committed as {git_current_hash()}")
        mutation_adopted = True
    else:
        log("\\nControl strategy defended its position (or all lost). No mutation adopted.")

    return mutation_adopted

"""

main_start = content.find("def main():")
content = content[:run_gen_start] + new_run_gen + "\n" + content[main_start:]

# 3. Patch main loop to call run_tournament instead of run_generation
content = content.replace("run_generation(client, generation, temperature=current_temp)", "run_tournament(client, generation, temperature=current_temp)")

with open('autoresearch.py', 'w') as f:
    f.write(content)

