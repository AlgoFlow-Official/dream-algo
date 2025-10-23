import os
import glob
import pandas as pd

WATCHLIST_DIR = "data/watchlists"
MASTER_FILE = "data/master_watchlist.csv"

def update_master_log():
    # Make sure the directory exists
    os.makedirs(WATCHLIST_DIR, exist_ok=True)

    # Find all daily watchlists like data/watchlists/watchlist_YYYYMMDD.csv
    files = sorted(glob.glob(os.path.join(WATCHLIST_DIR, "watchlist_*.csv")))
    if not files:
        print(f"No watchlist files found in {WATCHLIST_DIR}")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)

            # If "date" is missing, infer it from the filename
            if "date" not in df.columns:
                base = os.path.basename(f)
                inferred_date = base.replace("watchlist_", "").replace(".csv", "")
                df["date"] = inferred_date

            # Keep only non-empty frames
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        print("No valid watchlists to merge.")
        return

    master_df = pd.concat(dfs, ignore_index=True)

    # Sort & de-dup by date+ticker so last write wins
    if "ticker" in master_df.columns:
        master_df.sort_values(["date", "ticker"], inplace=True)
        master_df.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)

    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    master_df.to_csv(MASTER_FILE, index=False)
    print(f"[+] Master log updated → {MASTER_FILE} (rows={len(master_df)})")

if __name__ == "__main__":
    update_master_log()
