import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/dataset_intraday.csv")

print("\n📊 First 10 rows of the dataset:")
print(df.head(10))

# --- Pops analysis ---
pops = df.groupby("ticker")["label_pop5"].sum().sort_values(ascending=False)
print("\n🔥 Number of 5% Pops per ticker:")
print(pops)

# Plot pops
pops.plot(kind="bar", title="5% Pops per Ticker")
plt.show()

# --- Premarket gap vs return ---
print("\n📈 Checking relationship between premarket gap and intraday return...")
df.plot.scatter(x="premarket_gap_pct", y="intraday_return_pct", alpha=0.6)
plt.title("Premarket Gap vs Intraday Return")
plt.show()

# --- Tomorrow’s Watchlist ---
watchlist = (
    df.sort_values(by=["premarket_gap_pct", "premarket_volume"], ascending=False)
    .head(5)[["date", "ticker", "premarket_gap_pct", "premarket_volume"]]
)

print("\n🚀 Tomorrow’s Watchlist (Top 5 Gappers):")
print(watchlist)

# Save to CSV
import os
os.makedirs("data", exist_ok=True)
from datetime import datetime
import os

os.makedirs("data/watchlists", exist_ok=True)

# filename includes today's date (YYYYMMDD format)
date_str = datetime.now().strftime("%Y%m%d")
watchlist_file = f"data/watchlists/watchlist_{date_str}.csv"

watchlist.to_csv(watchlist_file, index=False)
print(f"\n✅ Watchlist saved to {watchlist_file}")

# Update the master log automatically
try:
    import update_master_log
    update_master_log.update_master_log()
    print("✅ Master log updated.")
except Exception as e:
    print(f"⚠️ Master log update failed: {e}")

# Open today’s watchlist in Numbers
try:
    os.system(f'open -a Numbers "{watchlist_file}"')
    print("📂 Opened today’s watchlist in Numbers.")
except Exception as e:
    print(f"⚠️ Could not open watchlist automatically: {e}")


# Update the master log automatically
try:
    import update_master_log
    update_master_log.update_master_log()
    print("✅ Master log updated.")
except Exception as e:
    print(f"⚠️ Master log update failed: {e}")
