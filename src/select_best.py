#!/usr/bin/env python3
import json
import os
import subprocess

BEST_PARAMS_JSON = "data/best_params.json"
EQUITY_PLOT = "data/backtest_equity.png"
TRADES_DIR = "data"

def run_best():
    if not os.path.exists(BEST_PARAMS_JSON):
        raise FileNotFoundError(f"{BEST_PARAMS_JSON} not found. Run select_best.py first.")

    # Load best params
    with open(BEST_PARAMS_JSON, "r") as f:
        params = json.load(f)

    strategy = params.get("strategy")
    if not strategy:
        raise ValueError("No strategy found in best_params.json")

    print(f"[INFO] Running backtest with strategy = {strategy}\n")

    # Call backtest script with --strategy arg
    cmd = ["python", "src/backtest.py", "--strategy", strategy]
    subprocess.run(cmd)

    # Open equity plot if it exists
    if os.path.exists(EQUITY_PLOT):
        print(f"\n[INFO] Opening equity plot -> {EQUITY_PLOT}\n")
        try:
            subprocess.run(["open", EQUITY_PLOT])  # macOS
        except Exception:
            try:
                subprocess.run(["xdg-open", EQUITY_PLOT])  # Linux
            except Exception:
                print("[WARN] Could not auto-open equity plot. Please open manually.")

    # Try to open trades CSV
    trades_file = os.path.join(TRADES_DIR, f"trades_{strategy}.csv")
    if os.path.exists(trades_file):
        print(f"[INFO] Opening trades file -> {trades_file}\n")
        try:
            subprocess.run(["open", trades_file])  # macOS
        except Exception:
            try:
                subprocess.run(["xdg-open", trades_file])  # Linux
            except Exception:
                print("[WARN] Could not auto-open trades CSV. Please open manually.")
    else:
        print(f"[WARN] Trades file not found: {trades_file}")

if __name__ == "__main__":
    run_best()
