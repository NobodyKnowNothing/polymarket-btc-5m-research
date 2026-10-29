"""
Position-size calculator for Polymarket BTC 5-minute market trades.

Given a dollar amount and outcome, fetches the live order book,
walks the asks to simulate a realistic fill (accounting for depth),
and applies Polymarket's exact taker fee formula to return the
number of shares you'd actually receive.

Fee formula (crypto category):
    fee_per_share = CATEGORY_RATE * fill_price * (1 - fill_price)
    where CATEGORY_RATE = 0.072  (peaks at 1.80% when p = 0.50)
"""

import json
import requests
from main import PolymarketBTC5m

# Polymarket crypto category rate.
# fee_per_share = CRYPTO_CATEGORY_RATE * p * (1 - p)
# At p=0.50 this yields a 1.80% fee (the published peak for crypto).
CRYPTO_CATEGORY_RATE = 0.072

CLOB_BOOK_URL = "https://clob.polymarket.com/book"
HEADERS = {"Accept": "application/json"}


def _fee_per_share(price: float) -> float:
    """Polymarket taker fee for one share at a given fill price."""
    return CRYPTO_CATEGORY_RATE * price * (1.0 - price)


def _walk_book(asks: list[dict], dollars: float) -> tuple[float, float]:
    """
    Walk the ask side of the order book to fill a dollar-denominated buy.

    For each price level, computes the effective cost per share
    (price + fee) and fills as many shares as the remaining budget
    and available size allow.

    Returns:
        (total_shares, dollars_spent)
    """
    remaining = dollars
    total_shares = 0.0

    for level in asks:
        price = float(level["price"])
        size = float(level["size"])
        fee = _fee_per_share(price)
        cost_per_share = price + fee

        affordable = remaining / cost_per_share
        fill = min(affordable, size)

        total_shares += fill
        remaining -= fill * cost_per_share

        if remaining <= 1e-9:
            break

    return total_shares, dollars - remaining


def position_size(
    dollars: float,
    outcome: str = "Up",
    market: PolymarketBTC5m | None = None,
) -> tuple[float, float]:
    """
    Return the number of shares received for a dollar amount after
    slippage (order-book depth) and Polymarket taker fees.

    Args:
        dollars:  Amount to spend in USD.
        outcome:  "Up" or "Down".
        market:   Existing PolymarketBTC5m instance (creates one if None).

    Returns:
        (shares, unfilled_dollars)
        shares          — number of shares acquired.
        unfilled_dollars — amount left unspent if the book was too thin.
                           0.0 means the order was fully filled.
    """
    if market is None:
        market = PolymarketBTC5m()

    # Resolve the token id for this outcome
    market._ensure_metadata()
    token_id = market._metadata.get(outcome)
    if token_id is None:
        raise ValueError(
            f"Outcome '{outcome}' not found. "
            f"Available: {list(market._metadata.keys())}"
        )

    # Fetch the full order book
    try:
        res = requests.get(
            f"{CLOB_BOOK_URL}?token_id={token_id}",
            headers=HEADERS,
            timeout=5,
        )
        res.raise_for_status()
        book = res.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch order book: {e}")

    asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
    if not asks:
        return 0.0, dollars

    shares, spent = _walk_book(asks, dollars)
    unfilled = round(dollars - spent, 6)
    return round(shares, 6), unfilled


def book_depth_usd(asks: list[dict]) -> float:
    """Total dollar value of resting ask-side liquidity."""
    return sum(float(a["price"]) * float(a["size"]) for a in asks)


def position_size_detailed(
    dollars: float,
    outcome: str = "Up",
    market: PolymarketBTC5m | None = None,
) -> dict:
    """
    Enhanced position sizing with fidelity metrics.

    Returns dict with:
        shares          - number of shares acquired
        unfilled        - dollars left unspent
        avg_fill_price  - effective cost per share (price + fee, weighted)
        book_depth_usd  - total dollar depth on ask side
        best_ask        - best (lowest) ask price
        spread          - best_ask - best_bid
    """
    empty = {
        "shares": 0.0, "unfilled": dollars, "avg_fill_price": 0.0,
        "book_depth_usd": 0.0, "best_ask": 0.0, "spread": 0.0,
    }

    if market is None:
        market = PolymarketBTC5m()

    market._ensure_metadata()
    token_id = market._metadata.get(outcome)
    if token_id is None:
        return empty

    try:
        res = requests.get(
            f"{CLOB_BOOK_URL}?token_id={token_id}",
            headers=HEADERS,
            timeout=5,
        )
        res.raise_for_status()
        book = res.json()
    except Exception:
        return empty

    asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
    bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)

    if not asks:
        return empty

    shares, spent = _walk_book(asks, dollars)
    depth = book_depth_usd(asks)
    best_ask = float(asks[0]["price"]) if asks else 0.0
    best_bid = float(bids[0]["price"]) if bids else 0.0
    spread = round(best_ask - best_bid, 4) if best_ask and best_bid else 0.0
    avg_fill = round(spent / shares, 6) if shares > 0 else 0.0

    return {
        "shares": round(shares, 6),
        "unfilled": round(dollars - spent, 6),
        "avg_fill_price": avg_fill,
        "book_depth_usd": round(depth, 2),
        "best_ask": best_ask,
        "spread": spread,
    }


if __name__ == "__main__":
    market = PolymarketBTC5m()

    for side in ("Up", "Down"):
        fill = position_size_detailed(dollars=10.0, outcome=side, market=market)
        print(f"{side}: {fill['shares']} shares for $10.00 "
              f"(${fill['unfilled']:.2f} unfilled, "
              f"avg_fill=${fill['avg_fill_price']:.4f}, "
              f"depth=${fill['book_depth_usd']:.2f}, "
              f"spread={fill['spread']:.4f})")
