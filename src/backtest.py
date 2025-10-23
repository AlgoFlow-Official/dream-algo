import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

# === Helpers ===
def load_predictions(strategy):
    pred_file = f"data/predictions_{strategy}.csv"
    if not os.path.exists(pred_file):
        raise ValueError(f"Strategy {strategy} not found in predictions CSV")

    df = pd.read_csv(pred_file)

    # If Return column missing, generate dummy one from True_Label vs prediction
    if "Return" not in df.columns:
        if "True_Label" in df.columns and strategy in df.columns:
            print("[WARN] No 'Return' column found. Creating dummy returns (1 if correct, -1 if wrong).")
            df["Return"] = (df[strategy] == df["True_Label"]).astype(int) * 2 - 1
        else:
            raise ValueError(
                "Cannot generate 'Return' column: missing True_Label or strategy column."
            )

    return df

def save_equity_plot(df, strategy):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Equity"], label="Equity Curve")
    plt.title(f"Equity Curve - {strategy}")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.legend()
    out_path = "data/backtest_equity.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved equity plot -> {out_path}")

def save_trades(df, strategy):
    out_path = f"data/trades_{strategy}.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved trades -> {out_path}")

# === Core Backtest ===
def run_backtest(args):
    df = load_predictions(args.strategy)

    # Ensure safe default size
    if getattr(args, "size", None) is None:
        args.size = 1.0

    # Calculate equity curve
    df["Equity"] = (1 + df["Return"] * args.size).cumprod()

    # Save results
    save_equity_plot(df, args.strategy)
    save_trades(df, args.strategy)

    # Print summary metrics
    print("\n[BACKTEST RESULTS]")
    print(f"Strategy: {args.strategy}")
    print(f"Final Equity: {df['Equity'].iloc[-1]:.2f}")
    print(f"Trades: {len(df)}")

    return {
        "Strategy": args.strategy,
        "FinalEquity": df["Equity"].iloc[-1],
        "Trades": len(df),
    }

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name")
    parser.add_argument("--size", type=float, default=None, help="Position size multiplier")
    args = parser.parse_args()

    run_backtest(args)
