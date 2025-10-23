# src/universal_scanner.py
"""
Dream Algo v1.5 — Universal Market Scanner + AI Signal
------------------------------------------------------
Scans Yahoo Finance “Most Active” tickers, fetches real prices,
generates AI-style confidence, direction, and reasoning.
"""

import os, io, requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import random

def universal_scan(limit: int = 100) -> pd.DataFrame:
    # --- ensure /data exists ---
    save_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "live_signals.csv")

    # --- fetch tickers ---
    url = "https://finance.yahoo.com/most-active"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
    tables = pd.read_html(io.StringIO(html))
    tickers = tables[0]["Symbol"].head(limit).tolist()

    results = []
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            fi = getattr(t, "fast_info", {})
            price = fi.get("last_price")
            if not price or price == 0:
                hist = t.history(period="1d")
                price = hist["Close"].iloc[-1] if not hist.empty else 0

            vol = fi.get("last_volume") or 0
            avg_vol = fi.get("ten_day_average_volume") or 1
            rel_vol = round(vol / avg_vol, 2) if avg_vol else 0
            change = round(((fi.get("last_price") - fi.get("previous_close", 0)) /
                            fi.get("previous_close", 1)) * 100, 2) if fi.get("previous_close") else 0

            # --- simple AI scoring logic ---
            ai_score = (abs(change) * 0.4) + (rel_vol * 0.6) + random.uniform(0, 5)
            if ai_score > 20:
                signal = "BUY"
            elif ai_score < 5:
                signal = "SELL"
            else:
                signal = "WATCH"
            confidence = min(round(ai_score * 4, 1), 99.9)

            if signal == "BUY":
                reason = "Momentum surge + volume breakout potential."
            elif signal == "SELL":
                reason = "Low momentum / volume tapering detected."
            else:
                reason = "Neutral — awaiting confirmation of trend."

            results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": tk,
                "price": round(price, 2),
                "%_change": change,
                "volume": int(vol),
                "rel_volume": rel_vol,
                "signal": signal,
                "confidence": confidence,
                "reason": reason
            })
        except Exception as e:
            print(f"⚠️ {tk} skipped: {e}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values(by=["confidence"], ascending=False)
    df.to_csv(save_path, index=False)
    print(f"✅ AI scan complete — saved {len(df)} tickers to {save_path}")
    return df


if __name__ == "__main__":
    universal_scan(50)
