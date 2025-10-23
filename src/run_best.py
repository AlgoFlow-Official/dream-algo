"""
run_best.py — Dream Algo (Stable Bypass + Logger Edition)
---------------------------------------------------------
Lightweight, production-safe build.
Generates live signals directly from market + model output,
skips backtest, and logs each run for scheduler integration.
"""

import os
import json
import datetime
import pandas as pd
import csv
import time

from utils_market import fetch_premarket_data

# ─────────────────────────────
#  Config paths
# ─────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
LOG_DIR = os.path.join(BASE_DIR, "../logs")
BEST_PARAMS_PATH = os.path.join(DATA_DIR, "best_params.json")
PRED_FILE = os.path.join(DATA_DIR, "predictions_Pred_Tuned.csv")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "dream_runs.csv")

# ─────────────────────────────
#  Helpers
# ─────────────────────────────
def load_best_params():
    """Return saved best parameters or default."""
    try:
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"strategy": "Pred_Tuned"}

# ─────────────────────────────
#  Core Dream Algo
# ─────────────────────────────
def run_dream_algo():
    """Generate live signals quickly (bypass backtest)."""
    print("🚀 Running Dream Algo (Stable Bypass + Logger Edition)")

    params = load_best_params()
    strategy = params.get("strategy", "Pred_Tuned")

    # 1️⃣ Fetch market data
    try:
        df_market = fetch_premarket_data()
        if df_market is None or df_market.empty:
            raise ValueError("No market data returned.")
        print(f"📊 Market data loaded: {len(df_market)} tickers.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch market data: {e}")
        return []

    # 2️⃣ Load model output if available
    df_pred = None
    if os.path.exists(PRED_FILE):
        try:
            df_pred = pd.read_csv(PRED_FILE)
            print(f"[INFO] Loaded model output: {len(df_pred)} rows.")
        except Exception as e:
            print(f"[WARN] Could not read predictions file: {e}")

    # 3️⃣ Combine data → quick signals
    rows, now = [], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, row in df_market.iterrows():
        conf, sig = 0.0, "hold"
        if df_pred is not None and i < len(df_pred):
            prob = float(df_pred.loc[i, "Prob"])
            pred = int(df_pred.loc[i, "Pred_Tuned"])
            conf = round(prob * 100, 1)
            sig = "buy" if pred == 1 else "hold"

        rows.append({
            "timestamp": now,
            "ticker": row.get("Ticker", row.get("Symbol", "N/A")),
            "price": float(row.get("Price", 0)),
            "change_pct": float(row.get("ChangePct", 0)),
            "volume": int(row.get("Volume", 0)),
            "signal": sig,
            "confidence": conf
        })

    print(f"✅ Generated {len(rows)} quick signals.")
    return rows

# ─────────────────────────────
#  Manual / Scheduler Run
# ─────────────────────────────
if __name__ == "__main__":
    start = time.time()
    data = run_dream_algo()
    runtime = round(time.time() - start, 2)

    if data:
        df_out = pd.DataFrame(data)
        print("\nSample output:")
        print(df_out.head())

        # 🔹 Log summary
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["timestamp", "signals", "runtime_sec"])
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    len(data),
                    runtime
                ])
            print(f"[LOG] Saved run summary → {LOG_FILE}")
        except Exception as e:
            print(f"[WARN] Logging failed: {e}")
    else:
        print("❌ No signals generated.")
