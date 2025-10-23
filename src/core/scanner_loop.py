# src/core/scanner_loop.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio, json, pandas as pd
from datetime import datetime, time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
from providers.alpaca_prices import fetch_snapshots as get_market_data


# ===== CONFIGURATION =====
CFG = {
    "timezone": "America/Toronto",
    "scan_hours": {"start": "04:00", "end": "20:00"}
}

# ===== SIGNAL & RANKING ENGINE =====
def compute_signals(df: pd.DataFrame):
    if df.empty:
        print("[WARN] No data received from Alpaca.")
        return df
    df["gap_pct"] = df.apply(
        lambda x: ((x["price"] - x["prev_close"]) / x["prev_close"] * 100)
        if x["price"] and x["prev_close"] else 0,
        axis=1
    )
    df["rvol_z"] = (df["volume"] - df["volume"].mean()) / (df["volume"].std() + 1e-9)
    df["hype_score"] = 0.6 * df["gap_pct"] + 0.4 * df["rvol_z"]
    df = df.sort_values("hype_score", ascending=False)
    return df

# ===== MAIN SCANNER LOOP =====
async def scan_once():
    df = get_market_data()
    df = compute_signals(df)
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[SCAN @ {now}]")
    if df.empty:
        print("No tickers returned. Check API connection or keys.")
    else:
        print(df[["ticker", "price", "gap_pct", "rvol_z", "hype_score"]])

async def runner():
    tz = pytz.timezone(CFG["timezone"])
    start = time.fromisoformat(CFG["scan_hours"]["start"])
    end = time.fromisoformat(CFG["scan_hours"]["end"])

    scheduler = AsyncIOScheduler(timezone=str(tz))
    scheduler.add_job(scan_once, "interval", seconds=30)  # every 30s for now
    scheduler.start()

    while True:
        now = datetime.now(tz).time()
        await asyncio.sleep(1 if (start <= now <= end) else 60)

if __name__ == "__main__":
    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        print("Scanner stopped manually.")
