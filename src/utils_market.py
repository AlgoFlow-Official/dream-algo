from datetime import datetime, time
import pytz

NY = pytz.timezone("America/New_York")

PRE_START = time(4, 0)    # 04:00
PRE_END   = time(9, 29)   # 09:29
REG_START = time(9, 30)   # 09:30
REG_END   = time(16, 0)   # 16:00

def to_ny(ts):
    """Ensure a pandas DateTimeIndex or Timestamp is in America/New_York."""
    if getattr(ts, "tz", None) is None:
        return ts.tz_localize("UTC").tz_convert(NY)
    return ts.tz_convert(NY)
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_premarket_data(limit=25):
    """
    Fetches top tickers (gainers/active) for premarket or live session.
    Returns a DataFrame with columns: Ticker, Price, ChangePct, Volume.
    """

    print("📊 Fetching premarket data...")

    # Define a few popular tickers or sources to scan — can be expanded later
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOG", "AMD", "NFLX", "PLTR",
               "SOFI", "RIVN", "BABA", "INTC", "SNAP", "PYPL", "NIO", "COIN", "SHOP", "UBER"]

    data = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="2d", interval="1h")
            if hist.empty:
                continue

            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else last
            change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100

            data.append({
                "Ticker": t,
                "Price": round(last["Close"], 2),
                "ChangePct": round(change_pct, 2),
                "Volume": int(last["Volume"]),
                "Signal": "buy" if change_pct > 0.3 else "sell" if change_pct < -0.3 else "hold",
                "Confidence": abs(round(change_pct * 2, 1))  # fake confidence metric
            })
        except Exception as e:
            print(f"[WARN] Failed to fetch {t}: {e}")
            continue

    df = pd.DataFrame(data)
    df = df.sort_values(by="ChangePct", ascending=False).head(limit)
    print(f"✅ Loaded {len(df)} tickers.")
    return df
