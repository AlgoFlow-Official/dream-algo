import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, UTC

# ==========================================================
# Integration with your model pipeline (src/predict.py)
# This code ADAPTS automatically to whatever predict() returns:
# - DataFrame with prob + OHLC columns  -> uses directly
# - DataFrame with prob only            -> merges with load_data()
# - Tuple (DataFrame, np.ndarray)       -> attaches probs, then merge if needed
# - np.ndarray (probs)                  -> attaches to load_data() rows
# ==========================================================
def _safe_import_predict_pipeline():
    """
    Import predict.py functions if available. Returns a dict of callables, some may be None.
    """
    try:
        from predict import predict as model_predict  # main entry
    except Exception:
        model_predict = None

    try:
        from predict import load_data as model_load_data  # to fetch dataset with OHLC
    except Exception:
        model_load_data = None

    return {"predict": model_predict, "load_data": model_load_data}

def load_predictions(strategy_name: str):
    """
    Merge model predictions with OHLC data from dataset_intraday.csv.
    Returns list of dicts with: date, price, high, low, close, prob
    """
    preds = pd.read_csv("data/test_predictions_xgb.csv", parse_dates=["Datetime"])
    data = pd.read_csv("data/dataset_intraday.csv", parse_dates=["Datetime"])

    # Merge predictions onto dataset by timestamp
    df = pd.merge(data, preds[["Datetime", "Prob"]], on="Datetime", how="inner")

    # Standardize column names for backtester
    df = df.rename(columns={
        "Datetime": "date",
        "Open": "price",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Prob": "prob"
    })

    # Keep only required columns
    return df[["date", "price", "high", "low", "close", "prob"]].to_dict("records")


    # Normalize column names
    rename_map = {
        "Datetime": "date",
        "Open": "price",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    if "probs" in df.columns:
        rename_map["probs"] = "prob"
    if "Pred_Prob" in df.columns:
        rename_map["Pred_Prob"] = "prob"

    df = df.rename(columns=rename_map)

    # Fallback: if prob still missing, derive from preds_default
    if "prob" not in df.columns and "preds_default" in df.columns:
        df["prob"] = (df["preds_default"].astype(float) + 0.5) / 2.0

    df = df[["date", "price", "high", "low", "close", "prob"]]
    return df.to_dict("records")


    # Helper to normalize/rename columns
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        if "Datetime" in df.columns:
            rename_map["Datetime"] = "date"
        # price source: Open preferred
        if "Open" in df.columns:
            rename_map["Open"] = "price"
        elif "price" not in df.columns and "open" in df.columns:
            rename_map["open"] = "price"
        # OHLC
        if "High" in df.columns:
            rename_map["High"] = "high"
        if "Low" in df.columns:
            rename_map["Low"] = "low"
        if "Close" in df.columns:
            rename_map["Close"] = "close"
        # probability
        if "probs" in df.columns and "prob" not in df.columns:
            rename_map["probs"] = "prob"
        if "Pred_Prob" in df.columns and "prob" not in df.columns:
            rename_map["Pred_Prob"] = "prob"
        if "prediction_proba" in df.columns and "prob" not in df.columns:
            rename_map["prediction_proba"] = "prob"

        df = df.rename(columns=rename_map)

        # If price missing but close exists, use close as price
        if "price" not in df.columns and "close" in df.columns:
            df["price"] = df["close"]

        return df

    # Case A: DataFrame returned
    if isinstance(out, pd.DataFrame):
        df_pred = _normalize_columns(out.copy())

        # If OHLC missing, try to merge with dataset
        need_ohlc = not {"price", "high", "low", "close"}.issubset(df_pred.columns)
        if need_ohlc:
            if model_load_data is None:
                raise RuntimeError(
                    "predict.predict() did not return OHLC columns and predict.load_data() is unavailable "
                    "to merge them. Update predict.py or return a DataFrame with OHLC + 'prob'."
                )
            df_data = model_load_data()
            df_data = _normalize_columns(df_data.copy())
            # Merge on date
            if "date" not in df_pred.columns and "date" in df_data.columns and "date" in out.columns:
                # fallback if rename didn't hit; attempt to coerce
                pass
            if "date" not in df_pred.columns and "Datetime" in out.columns:
                df_pred["date"] = pd.to_datetime(out["Datetime"])
            if "date" not in df_data.columns and "Datetime" in df_data.columns:
                df_data["date"] = pd.to_datetime(df_data["Datetime"])

            df_pred["date"] = pd.to_datetime(df_pred["date"])
            df_data["date"] = pd.to_datetime(df_data["date"])

            df = pd.merge(df_pred, df_data[["date", "price", "high", "low", "close"]], on="date", how="left")
        else:
            df = df_pred

        required = {"date", "price", "high", "low", "close"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise RuntimeError(f"Missing columns after normalization/merge: {missing}")

        # Ensure prob exists
        if "prob" not in df.columns:
            # If there is a binary prediction, make a neutral prob
            if "pred" in df.columns:
                df["prob"] = (df["pred"].astype(float) + 0.5) / 2.0
            else:
                raise RuntimeError("No 'prob' column found or derivable from predict() output.")

        # Final shaping
        df = df[["date", "price", "high", "low", "close", "prob"]].copy()
        return df.to_dict("records")

    # Case B: Tuple (maybe (df, probs) or (probs, df))
    if isinstance(out, tuple) and len(out) == 2:
        a, b = out
        if isinstance(a, pd.DataFrame) and isinstance(b, (np.ndarray, list, pd.Series)):
            df_base = _normalize_columns(a.copy())
            probs = np.asarray(b).reshape(-1)
        elif isinstance(b, pd.DataFrame) and isinstance(a, (np.ndarray, list, pd.Series)):
            df_base = _normalize_columns(b.copy())
            probs = np.asarray(a).reshape(-1)
        else:
            raise RuntimeError(
                "predict.predict() returned a tuple, but not in a supported shape: (DataFrame, probs) or (probs, DataFrame)."
            )

        if "prob" not in df_base.columns:
            if len(df_base) != len(probs):
                raise RuntimeError("Length mismatch between DataFrame rows and probs array.")
            df_base["prob"] = probs

        # Ensure OHLC; merge if missing
        if not {"price", "high", "low", "close"}.issubset(df_base.columns):
            if model_load_data is None:
                raise RuntimeError(
                    "OHLC columns missing and predict.load_data() unavailable to merge them."
                )
            df_data = _normalize_columns(model_load_data().copy())

            if "date" not in df_base.columns and "Datetime" in df_base.columns:
                df_base["date"] = pd.to_datetime(df_base["Datetime"])
            if "date" not in df_data.columns and "Datetime" in df_data.columns:
                df_data["date"] = pd.to_datetime(df_data["Datetime"])

            df_base["date"] = pd.to_datetime(df_base["date"])
            df_data["date"] = pd.to_datetime(df_data["date"])

            df_base = pd.merge(df_base, df_data[["date", "price", "high", "low", "close"]], on="date", how="left")

        df_base = df_base[["date", "price", "high", "low", "close", "prob"]].copy()
        return df_base.to_dict("records")

    # Case C: Only probs returned (np.ndarray / list / Series)
    if isinstance(out, (np.ndarray, list, pd.Series)):
        if model_load_data is None:
            raise RuntimeError(
                "predict.predict() returned only probabilities, but predict.load_data() is not available to attach OHLC."
            )
        probs = np.asarray(out).reshape(-1)
        df_data = _normalize_columns(model_load_data().copy())

        if "date" not in df_data.columns and "Datetime" in df_data.columns:
            df_data["date"] = pd.to_datetime(df_data["Datetime"])

        if len(df_data) != len(probs):
            raise RuntimeError("Length mismatch: dataset rows and returned probs differ.")

        df_data["prob"] = probs
        df_data = df_data[["date", "price", "high", "low", "close", "prob"]].copy()
        return df_data.to_dict("records")

    raise RuntimeError(
        "Unsupported return type from predict.predict(). "
        "Return a DataFrame with columns (Datetime/Open/High/Low/Close and prob), "
        "or (DataFrame, probs), or just probs (with load_data available)."
    )


# ==========================================================
# Trade Recorder and Utilities
# ==========================================================
equity_curve = [1.0]
trades = []

def record_trade(label, pnl, date, meta=None):
    global equity_curve, trades
    equity_curve.append(equity_curve[-1] * (1 + pnl))
    rec = {"ExitType": label, "PnL": pnl, "Date": date}
    if meta:
        rec.update(meta)
    trades.append(rec)


# ==========================================================
# Metrics
# ==========================================================
def _max_drawdown(equity: list[float]) -> float:
    eq = np.array(equity, dtype=float)
    roll_max = np.maximum.accumulate(eq)
    drawdowns = (eq - roll_max) / roll_max
    return float(-np.min(drawdowns)) * 100.0  # percent

def sharpe_ratio(r):
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.mean(r) / (np.std(r) + 1e-12))

def sortino_ratio(r):
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 0.0
    downside = r[r < 0]
    denom = np.std(downside) if downside.size else 0.0
    return float(np.mean(r) / (denom + 1e-12))

def profit_factor(r):
    r = np.asarray(r, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


# ==========================================================
# Backtest
# ==========================================================
def run_backtest(args):
    global trades, equity_curve
    trades = []
    equity_curve = [1.0]
    open_trades = []
    daily_count = {}

    predictions = load_predictions(args.strategy)

    for pred in predictions:
        date = pd.to_datetime(pred["date"])
        price = float(pred["price"])
        high = float(pred["high"])
        low = float(pred["low"])
        close = float(pred["close"])
        prob = float(pred.get("prob", 1.0))

        # ---- Filters ----
        if prob < args.prob_threshold:
            continue

        dkey = date.date()
        if args.daily_limit is not None:
            if daily_count.get(dkey, 0) >= args.daily_limit:
                continue
            daily_count[dkey] = daily_count.get(dkey, 0) + 1

        if args.max_trades is not None and len(open_trades) >= args.max_trades:
            continue

        # ---- Open trade ----
        entry_price = price
        stop_price = entry_price * (1 - args.stop) if args.stop is not None else None
        take_price = entry_price * (1 + args.take) if args.take is not None else None
        trail_price = None

        position = {
            "entry": entry_price,
            "stop": stop_price,
            "take": take_price,
            "trail": trail_price,
            "size": float(args.size),
            "date": date,
            "prob": prob,
        }
        open_trades.append(position)

        # ---- Exit logic (1-bar hold model) ----
        # Trailing stop (updated from current bar's high)
        if args.trail is not None:
            if position["trail"] is None:
                position["trail"] = entry_price * (1 - args.trail)
            else:
                new_trail = high * (1 - args.trail)
                position["trail"] = max(position["trail"], new_trail)

        # Hard stop
        if position["stop"] is not None and low <= position["stop"]:
            pnl = (position["stop"] - entry_price) / entry_price * position["size"]
            record_trade("STOP", pnl, date, {"prob": prob})
            open_trades.remove(position)
            continue

        # Take profit
        if position["take"] is not None and high >= position["take"]:
            pnl = (position["take"] - entry_price) / entry_price * position["size"]
            record_trade("TAKE", pnl, date, {"prob": prob})
            open_trades.remove(position)
            continue

        # Trailing stop hit
        if position["trail"] is not None and low <= position["trail"]:
            pnl = (position["trail"] - entry_price) / entry_price * position["size"]
            record_trade("TRAIL", pnl, date, {"prob": prob})
            open_trades.remove(position)
            continue

        # Default: exit at close of the bar
        pnl = (close - entry_price) / entry_price * position["size"]
        record_trade("EXIT", pnl, date, {"prob": prob})
        open_trades.remove(position)

    # ---- Metrics ----
    df = pd.DataFrame(trades)
    if df.empty:
        returns = np.array([], dtype=float)
    else:
        returns = df["PnL"].astype(float).values

    metrics = {
        "Return%": (equity_curve[-1] - 1) * 100.0,
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "MaxDD%": _max_drawdown(equity_curve),
        "Trades": int(len(trades)),
        "WinRate%": float((df["PnL"] > 0).mean() * 100.0) if not df.empty else 0.0,
        "ProfitFactor": profit_factor(returns) if returns.size else 0.0,
        "Exposure%": 100.0 * (len(trades) / max(len(predictions), 1)),  # crude bar-level exposure
    }

    # ---- Output directories (timestamped run) ----
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data") / f"backtest_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Print & Save trades ----
    print("[BACKTEST RESULTS]")
    print(pd.DataFrame([metrics]))
    trades_path = run_dir / "trades_results.csv"
    df.to_csv(trades_path, index=False)

    # ---- Save equity curve ----
    plt.figure(figsize=(11, 5))
    plt.plot(equity_curve, label="Equity")
    # annotate special exits
    for i, row in df.iterrows():
        if row["ExitType"] in ("STOP", "TAKE", "TRAIL"):
            plt.scatter(i + 1, equity_curve[i + 1], s=16)  # +1 because equity_curve includes initial 1.0
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()
    eq_path = run_dir / "backtest_equity.png"
    plt.savefig(eq_path)
    plt.close()

    # ---- Append metrics to a master CSV (with header if new) ----
    metrics_row = metrics.copy()
    metrics_row["Timestamp"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    metrics_row["Strategy"] = args.strategy
    metrics_row["ProbThr"] = args.prob_threshold
    metrics_row["Size"] = args.size
    metrics_row["Stop"] = args.stop
    metrics_row["Take"] = args.take
    metrics_row["Trail"] = args.trail
    metrics_row["MaxTrades"] = args.max_trades
    metrics_row["DailyLimit"] = args.daily_limit
    metrics_row["RunDir"] = str(run_dir)

    master_path = Path("data") / "backtest_results.csv"
    write_header = not master_path.exists()
    pd.DataFrame([metrics_row]).to_csv(master_path, mode="a", header=write_header, index=False)

    print(f"[INFO] Saved trades -> {trades_path}")
    print(f"[INFO] Saved equity plot -> {eq_path}")
    print(f"[INFO] Appended metrics -> {master_path}")


# ==========================================================
# Entry point / Args
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        help="A label for the run; not used to load files anymore.")
    parser.add_argument("--prob-threshold", type=float, default=0.0,
                        help="Skip trades with model probability below this value.")
    parser.add_argument("--size", type=float, default=1.0,
                        help="Position size fraction per trade (1.0=100%, 0.25=25%).")
    parser.add_argument("--stop", type=float, default=None,
                        help="Stop loss percent, e.g., 0.01 for 1%.")
    parser.add_argument("--take", type=float, default=None,
                        help="Take profit percent, e.g., 0.03 for 3%.")
    parser.add_argument("--trail", type=float, default=None,
                        help="Trailing stop percent, e.g., 0.02 for 2%.")
    parser.add_argument("--max-trades", type=int, default=None,
                        help="Maximum concurrent trades.")
    parser.add_argument("--daily-limit", type=int, default=None,
                        help="Maximum number of trades per calendar day.")
    args = parser.parse_args()

    run_backtest(args)
