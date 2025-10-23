# src/demo_showcase.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from datetime import datetime
from src.providers.alpaca_prices import fetch_snapshots

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple metrics for the demo."""
    if df.empty:
        print("[WARN] No data returned.")
        return df
    df["rvol_z"] = ((df["volume"] - df["volume"].mean()) / df["volume"].std()).round(3)
    df["hype_score"] = (
        0.4 * df["gap_pct"] + 0.3 * df["rvol_z"] + 0.3 * (df["price"].pct_change().fillna(0) * 100)
    ).round(3)
    return df.sort_values("hype_score", ascending=False)

def run_demo():
    print("\n🔥 MY ALGO LIVE DEMO —", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Fetching real-time data...\n")
    df = fetch_snapshots()
    if df.empty:
        print("⚠️ No tickers returned — check API connection.")
        return
    df["gap_pct"] = ((df["price"] - df["prev_close"]) / df["prev_close"] * 100).round(2)
    df = compute_signals(df)
    print(df[["ticker", "price", "gap_pct", "rvol_z", "hype_score"]].head(10).to_string(index=False))
    print("\n✅ Demo complete — live data powered by MY ALGO.")

if __name__ == "__main__":
    run_demo()
