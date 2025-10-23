import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

DATA_PATH = "data/dataset_intraday.csv"

# ---------- Helpers ----------
def ensure_series(x):
    """Force any input into a flat 1D pandas Series"""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        else:
            return pd.Series(x.to_numpy().ravel())
    return pd.Series(x).squeeze()

def compute_rsi(series, window: int = 14) -> pd.Series:
    series = ensure_series(series).astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def compute_vwap(prices, volumes):
    prices = ensure_series(prices).astype(float)
    volumes = ensure_series(volumes).astype(float)
    cum_pv = (prices * volumes).cumsum()
    cum_v = volumes.cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def load_tickers() -> list:
    uni_txt = Path("data/universe.txt")
    if uni_txt.exists():
        return [t.strip().upper() for t in uni_txt.read_text().splitlines() if t.strip()]
    return ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

# ---------- Core ----------
def build_dataset(tickers: list, period: str = "10d", interval: str = "15m") -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    all_frames = []
    success, skipped = [], []

    for ticker in tickers:
        print(f"[INFO] Fetching {ticker} ({period}, {interval}) ...")
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                prepost=True,
                progress=False,
                threads=False
            )
            print(f"  -> got {len(df)} rows for {ticker}")

            if df.empty:
                print(f"[WARN] {ticker}: no data returned, skipping.")
                skipped.append(ticker)
                continue

            # --- Flatten columns if MultiIndex ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]
            # Standardize names
            rename_map = {c: c.split("_")[0] for c in df.columns}
            df.rename(columns=rename_map, inplace=True)

            # Force core cols into 1D Series
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df:
                    df[col] = ensure_series(df[col])

            # --- Feature Engineering ---
            df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
            df["volatility"] = (df["High"] - df["Low"]) / df["Open"]
            df["volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()

            # Relative Volume (≈2600 bars in 10d at 15m)
            df["rvol_10d"] = df["Volume"] / df["Volume"].rolling(2600, min_periods=20).mean()

            # VWAP + deviation
            vwap = compute_vwap(df["Close"], df["Volume"])
            df["vwap"] = vwap
            df["vwap_dev"] = (df["Close"] - vwap) / vwap

            # Momentum (5 bars = 75 minutes on 15m)
            df["momentum_5"] = df["Close"].pct_change(5)

            # RSI
            df["rsi_14"] = compute_rsi(df["Close"])

            # MACD
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            df["ticker"] = ticker

            # Clean up
            df = df.replace([np.inf, -np.inf], np.nan)
            if df.dropna().empty:
                print(f"[WARN] {ticker}: all rows empty after feature calc, skipping.")
                skipped.append(ticker)
                continue

            all_frames.append(df)
            success.append(ticker)

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            skipped.append(ticker)

    if not all_frames:
        print("[FATAL] No usable data. Try a different period/interval.")
        return pd.DataFrame()

    dataset = pd.concat(all_frames)
    dataset.reset_index(inplace=True)
    dataset.rename(columns={"index": "Datetime"}, inplace=True)

    dataset.to_csv(DATA_PATH, index=False)
    print(f"[SUCCESS] Saved dataset with shape {dataset.shape} -> {DATA_PATH}")

    # Summary
    print("\n[SUMMARY]")
    print("Success:", ", ".join(success) if success else "None")
    print("Skipped:", ", ".join(skipped) if skipped else "None")

    return dataset

# ---------- CLI ----------
if __name__ == "__main__":
    tickers = load_tickers()
    build_dataset(tickers)
