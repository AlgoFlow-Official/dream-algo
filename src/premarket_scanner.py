import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import os
from tabulate import tabulate

# Load tickers from file, or use defaults if missing
def load_tickers():
    path = "data/sample_tickers.txt"
    if os.path.exists(path):
        with open(path) as f:
            tickers = [line.strip() for line in f if line.strip()]
        if tickers:
            return tickers

    # Fallback default tickers
    print("[!] sample_tickers.txt not found or empty. Using default list.")
    return ["AAPL", "TSLA", "NVDA", "MSFT"]

# Fetch enriched premarket data for a single ticker
def fetch_premarket_data(ticker):
    try:
        tk = yf.Ticker(ticker)

        # Daily history (prev close, fundamentals)
        hist = tk.history(period="5d", interval="1d", prepost=True)
        if hist.empty:
            return None
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else hist["Close"].iloc[-1]
        last_close = hist["Close"].iloc[-1]
        open_price = hist["Open"].iloc[-1]

        # Intraday (1m) for premarket and volatility
        m1 = tk.history(period="1d", interval="1m", prepost=True)
        if m1.empty:
            return None
        premarket_price = m1["Close"].iloc[0]   # first premarket tick
        premarket_volume = m1["Volume"].iloc[0]

        # Features
        gap_pct = ((premarket_price - prev_close) / prev_close) * 100
        intraday_return = ((last_close - open_price) / open_price) * 100
        volatility = (m1["Close"].pct_change().std() or 0) * 100

        # Fundamentals
        info = tk.info
        market_cap = info.get("marketCap", np.nan)
        forward_pe = info.get("forwardPE", np.nan)
        beta = info.get("beta", np.nan)

        return {
            "ticker": ticker,
            "open": round(open_price, 4),
            "close": round(last_close, 4),
            "premarket_gap_pct": round(gap_pct, 4),
            "premarket_volume": premarket_volume,
            "intraday_return_pct": round(intraday_return, 4),
            "volatility_pct": round(volatility, 4),
            "market_cap": market_cap,
            "forward_PE": forward_pe,
            "beta": beta
        }
    except Exception:
        return None

# Main function
def main():
    tickers = load_tickers()
    results = []
    for t in tickers:
        data = fetch_premarket_data(t)
        if data:
            results.append(data)

    if not results:
        print("No pre-market data available.")
        return

    df = pd.DataFrame(results).sort_values(by="premarket_gap_pct", ascending=False)
    print(tabulate(df.head(15), headers="keys", tablefmt="psql"))

    # Save enriched CSV
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = f"logs/premarket_{timestamp}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n[+] Saved enriched premarket results to {out_file}")

if __name__ == "__main__":
    main()
