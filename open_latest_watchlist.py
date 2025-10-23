import os
import glob
from datetime import datetime

WATCHLIST_DIR = "data/watchlists"

def open_latest_watchlist():
    files = sorted(glob.glob(os.path.join(WATCHLIST_DIR, "watchlist_*.csv")))
    if not files:
        print("❌ No watchlist files found.")
        return

    latest_file = files[-1]
    print(f"✅ Latest watchlist: {latest_file}")

    # Try to open in Numbers (Mac only)
    os.system(f'open -a Numbers "{latest_file}"')

if __name__ == "__main__":
    open_latest_watchlist()
