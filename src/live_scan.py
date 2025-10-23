# src/live_scan.py
"""
Dream Algo — Universal Live Scanner (Phase 20 Final)
Scans the entire market for top gainers, losers, and most-active tickers.
Applies AI sentiment & buy/sell reasoning before saving to CSV.
"""

import requests
import pandas as pd
import time
from datetime import datetime
import pytz
from sentiment_engine import generate_ai_signals   # <- AI integration

MARKET_TZ = pytz.timezone("US/Eastern")
REFRESH_INTERVAL = 60   # seconds between updates

# Yahoo Finance public screener endpoints
SCREENER_ENDPOINTS = {
    "Top Gainers": "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=50&scrIds=day_gainers",
    "Top Losers":  "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=50&scrIds=day_losers",
    "Most Active": "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=50&scrIds=most_actives"
}


def fetch_yahoo_screeners() -> pd.DataFrame:
    """Pulls JSON screener data from Yahoo Finance and combines into one DataFrame."""
    all_data = []

    for category, url in SCREENER_ENDPOINTS.items():
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            quotes = r.json().get("finance", {}).get("result", [])[0].get("quotes", [])
            df = pd.DataFrame(quotes)
            if df.empty:
                continue
            df["category"] = category
            df = df[
                ["symbol", "shortName", "regularMarketPrice",
                 "regularMarketChangePercent", "regularMarketVolume", "marketCap", "category"]
            ]
            all_data.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {category}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined.rename(columns={
        "symbol": "ticker",
        "shortName": "name",
        "regularMarketPrice": "price",
        "regularMarketChangePercent": "pct_change",
        "regularMarketVolume": "volume",
        "marketCap": "market_cap"
    }, inplace=True)

    combined["timestamp"] = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S EST")
    return combined


def run_universal_scanner(save_path="data/live_movers.csv"):
    print("🚀 Dream Algo — Universal Live Scanner started.")
    while True:
        df = fetch_yahoo_screeners()
        if not df.empty:
            df = generate_ai_signals(df)     # <- Apply AI scoring
            df.to_csv(save_path, index=False)
            print(f"[{datetime.now(MARKET_TZ).strftime('%H:%M:%S')}] ✅ {len(df)} tickers scanned and saved.")
        else:
            print(f"[{datetime.now(MARKET_TZ).strftime('%H:%M:%S')}] ⚠️ No data received.")
        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    run_universal_scanner()
