import os
import subprocess
import pandas as pd
from itertools import product
from datetime import datetime

BACKTEST_RESULTS_CSV = os.path.join("data", "backtest_results.csv")

# ==========================================================
# Candidate parameter ranges
# ==========================================================
prob_thresholds = [0.55, 0.6, 0.65, 0.7]
sizes = [0.25, 0.5, 1.0]
stops = [0.005, 0.01, 0.02]
takes = [0.02, 0.03, 0.05]
trails = [0.01, 0.02]

# ==========================================================
# Run sweep
# ==========================================================
def run_sweep():
    results = []
    combos = list(product(prob_thresholds, sizes, stops, takes, trails))
    print(f"[INFO] Running parameter sweep ({len(combos)} combos)...")

    for prob, size, stop, take, trail in combos:
        cmd = [
            "python", "src/backtest.py",
            "--strategy", "Pred_Tuned",
            "--prob-threshold", str(prob),
            "--size", str(size),
            "--stop", str(stop),
            "--take", str(take),
            "--trail", str(trail),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if os.path.exists(BACKTEST_RESULTS_CSV):
                df = pd.read_csv(BACKTEST_RESULTS_CSV, on_bad_lines="skip")
                last = df.tail(1).to_dict("records")[0]
                last.update({
                    "prob": prob,
                    "size": size,
                    "stop": stop,
                    "take": take,
                    "trail": trail,
                })
                results.append(last)
                print(f"  ✓ prob={prob}, size={size}, stop={stop}, take={take}, trail={trail} | Sharpe={last.get('Sharpe'):.3f}")
            else:
                print(f"  ✗ prob={prob}, size={size}, stop={stop}, take={take}, trail={trail} | No results CSV")

        except subprocess.CalledProcessError as e:
            print(f"  ✗ prob={prob}, size={size}, stop={stop}, take={take}, trail={trail} | ERROR: {e.stderr.strip()}")

    if results:
        outdir = os.path.join("data", f"param_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "sweep_results.csv")
        pd.DataFrame(results).to_csv(outpath, index=False)
        print(f"[INFO] Sweep complete. Results saved -> {outpath}")
    else:
        print("[WARN] No results collected. Check backtest output format.")
    # --- Save best params ---
    try:
        best = df.sort_values("Sharpe", ascending=False).head(1)
        best_params_path = os.path.join(outdir, "best_params.json")
        best[["prob", "size", "stop", "take", "trail"]].to_json(
            best_params_path, orient="records", lines=False
        )
        print(f"[INFO] Best params saved -> {best_params_path}")
    except Exception as e:
        print(f"[WARN] Could not save best params: {e}")



if __name__ == "__main__":
    run_sweep()
