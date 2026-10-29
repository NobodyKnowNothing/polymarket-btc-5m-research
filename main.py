import time
import requests
import json
from datetime import datetime


class PolymarketBTC5m:
    """
    Tracks a rolling BTC up/down 5-minute market on Polymarket.

    Usage:
        market = PolymarketBTC5m()
        result = market.get_prices()   # single call — returns prices, spread, slippage

    The object automatically advances to the next market window once the
    current one expires.
    """

    GAMMA_URL = "https://gamma-api.polymarket.com/events"
    CLOB_BOOK_URL = "https://clob.polymarket.com/book"
    CLOB_MID_URL = "https://clob.polymarket.com/midpoint"
    HEADERS = {"Accept": "application/json"}
    INTERVAL = 300  # 5 minutes in seconds

    def __init__(self, initial_ts: int | None = None):
        """
        Args:
            initial_ts: Optional Unix timestamp for the market start.
                        If None, aligns to the current 5-minute boundary.
        """
        if initial_ts is not None:
            self.current_ts = initial_ts
        else:
            now = int(time.time())
            self.current_ts = now - (now % self.INTERVAL)

        # Will be populated lazily on the first price request
        self._metadata: dict[str, str] = {}  # {outcome_name: clob_token_id}
        self._metadata_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def slug(self) -> str:
        """The Gamma slug for the currently tracked market."""
        return f"btc-updown-5m-{self.current_ts}"

    @property
    def market_end_time(self) -> int:
        """Unix timestamp at which the current market window closes."""
        return self.current_ts + self.INTERVAL

    @property
    def is_expired(self) -> bool:
        """True when the system clock has passed the market end time."""
        return time.time() >= self.market_end_time

    def get_prices(self) -> dict:
        """
        Return the current state of the tracked market, including orderbook
        spread and buy-side slippage for each outcome.

        While the market is **live**, returns::

            {
                "status": "live",
                "slug": "btc-updown-5m-<ts>",
                "end_time": "HH:MM:SS",
                "prices": {
                    "Up": {
                        "midpoint": 0.xxx,
                        "best_bid": 0.xxx,
                        "best_ask": 0.xxx,
                        "spread": 0.xxx,
                        "slippage": 0.xxx
                    },
                    "Down": { ... }
                }
            }

        Once the market has **expired**, returns::

            {
                "status": "expired",
                "slug": "btc-updown-5m-<ts>",
                "resolved_at": "HH:MM:SS",
                "outcome": { <same structure as prices> },
                "new_slug": "btc-updown-5m-<next_ts>"
            }

        **spread**  = best_ask − best_bid  (width of the top-of-book gap)
        **slippage** = best_ask − midpoint  (extra cost to buy vs. theoretical mid)
        """
        # Ensure we have metadata for the current window
        self._ensure_metadata()

        if not self.is_expired:
            return self._build_response("live")
        else:
            return self._resolve_and_advance()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_metadata(self) -> None:
        """Fetch and cache outcome→token mapping from the Gamma API."""
        if self._metadata_loaded:
            return

        try:
            res = requests.get(
                f"{self.GAMMA_URL}?slug={self.slug}",
                headers=self.HEADERS,
                timeout=10,
            )
            data = res.json()

            if data and isinstance(data, list) and data[0].get("markets"):
                mkt = data[0]["markets"][0]

                raw_outcomes = mkt.get("outcomes", "[]")
                raw_clob_ids = mkt.get("clobTokenIds", "[]")
                outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else (raw_outcomes or [])
                clob_ids = json.loads(raw_clob_ids) if isinstance(raw_clob_ids, str) else (raw_clob_ids or [])

                if outcomes and clob_ids and len(outcomes) == len(clob_ids):
                    self._metadata = {
                        name: token_id
                        for name, token_id in zip(outcomes, clob_ids)
                        if name and token_id
                    }

                if self._metadata:
                    self._metadata_loaded = True

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Gamma API Error: {e}")

    def _fetch_book(self, token_id: str) -> dict:
        """
        Fetch the midpoint price and the full orderbook for a token.

        Uses /midpoint for the real price (Polymarket's internal calculation)
        and /book for the orderbook spread and buy-side slippage.

        Returns a dict with: midpoint, best_bid, best_ask, spread, slippage.
        All values default to 0.0 on error.
        """
        result = {
            "midpoint": 0.0,
            "best_bid": 0.0,
            "best_ask": 0.0,
            "spread": 0.0,
            "slippage": 0.0,
        }

        # --- 1. Midpoint (reliable price signal) ---
        try:
            mid_res = requests.get(
                f"{self.CLOB_MID_URL}?token_id={token_id}",
                headers=self.HEADERS,
                timeout=5,
            )
            if mid_res.status_code == 200:
                result["midpoint"] = float(mid_res.json().get("mid", 0.0))
        except Exception:
            pass

        # --- 2. Orderbook (spread + slippage) ---
        try:
            book_res = requests.get(
                f"{self.CLOB_BOOK_URL}?token_id={token_id}",
                headers=self.HEADERS,
                timeout=5,
            )
            if book_res.status_code == 200:
                book = book_res.json()
                bids = sorted(
                    book.get("bids", []),
                    key=lambda x: float(x["price"]),
                    reverse=True,
                )
                asks = sorted(
                    book.get("asks", []),
                    key=lambda x: float(x["price"]),
                )

                if bids:
                    result["best_bid"] = round(float(bids[0]["price"]), 4)
                if asks:
                    result["best_ask"] = round(float(asks[0]["price"]), 4)

                if bids and asks:
                    result["spread"] = round(
                        result["best_ask"] - result["best_bid"], 4
                    )
                    # Slippage: extra cost to buy vs. the midpoint
                    if result["midpoint"] > 0:
                        result["slippage"] = round(
                            result["best_ask"] - result["midpoint"], 4
                        )
        except Exception:
            pass

        return result

    def _fetch_all_books(self) -> dict:
        """Fetch orderbook metrics for every outcome in the current market."""
        prices: dict[str, dict] = {}
        for outcome_name, token_id in self._metadata.items():
            prices[outcome_name] = self._fetch_book(token_id)
        return prices

    def _build_response(self, status: str) -> dict:
        """Build a live-status response with current orderbook data."""
        return {
            "status": status,
            "slug": self.slug,
            "end_time": datetime.fromtimestamp(self.market_end_time).strftime("%H:%M:%S"),
            "prices": self._fetch_all_books(),
        }

    def _resolve_and_advance(self) -> dict:
        """
        Capture the final orderbook state for the expired market, then
        advance the internal timestamp to the next 5-minute window.
        """
        final_prices = self._fetch_all_books()
        old_slug = self.slug
        resolved_at = datetime.fromtimestamp(self.market_end_time).strftime("%H:%M:%S")

        # Advance to the next market window and reset metadata cache
        self.current_ts += self.INTERVAL
        self._metadata = {}
        self._metadata_loaded = False

        return {
            "status": "expired",
            "slug": old_slug,
            "resolved_at": resolved_at,
            "outcome": final_prices,
            "new_slug": self.slug,
        }


# ------------------------------------------------------------------
# Stream prices in a loop
# ------------------------------------------------------------------
if __name__ == "__main__":
    market = PolymarketBTC5m()

    while True:
        result = market.get_prices()
        print(json.dumps(result, indent=2))
        print()
        time.sleep(2)