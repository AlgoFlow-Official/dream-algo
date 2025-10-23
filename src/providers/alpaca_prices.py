# src/providers/alpaca_prices.py
import os, requests, pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://data.alpaca.markets/v2"
API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# Add or change your universe here
UNIVERSE = ["AAPL", "TSLA", "NVDA", "AMZN", "META"]

def fetch_snapshots():
    """Fetch snapshot data for multiple tickers."""
    results = []
    for symbol in UNIVERSE:
        try:
            r = requests.get(f"{BASE_URL}/stocks/{symbol}/snapshot", headers=HEADERS, timeout=10)
            if not r.ok:
                print(f"[WARN] {symbol} → HTTP {r.status_code}")
                continue
            j = r.json()
            last = j.get("latestTrade", {})
            prev = j.get("prevDailyBar", {})
            if not last or not prev:
                continue

            price = last.get("p")
            prev_close = prev.get("c")
            if not price or not prev_close:
                continue

            change = round(((price - prev_close) / prev_close) * 100, 2)
            vol = j.get("minuteBar", {}).get("v", 0)
            results.append({
                "ticker": symbol,
                "price": price,
                "prev_close": prev_close,
                "volume": vol,
                "change": change
            })
        except Exception as e:
            print(f"[WARN] Failed to get {symbol}: {e}")
    return pd.DataFrame(results)
