# ==============================================
# 💎 Dream Algo V1.7 — Universal AI Market Dashboard
# Real-time market analytics • 4 AM – 8 PM EST
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="Dream Algo Dashboard", layout="wide")
st.title("💎 Dream Algo V1.7 — Universal AI Market Dashboard")
st.caption("Real-time market analytics • 4 AM – 8 PM EST")

DATA_PATH = "data/live_movers.csv"
REFRESH_INTERVAL = 60  # seconds

# ------------------------------
# Load Data Safely
# ------------------------------
try:
    df = pd.read_csv(DATA_PATH)
    st.markdown(f"🕒 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")
    st.markdown(f"📁 **Rows Loaded:** {len(df)}")
except Exception as e:
    st.error(f"⚠️ Failed to load data: {e}")
    st.stop()

# ------------------------------
# Clean & Prepare Columns
# ------------------------------
df.columns = [c.strip().lower().replace("%", "pct").replace(" ", "_") for c in df.columns]

expected_cols = [
    "timestamp", "ticker", "sector", "price", "pct_change",
    "volume", "rel_volume", "signal", "confidence", "reason"
]

# Add missing columns with defaults (to prevent KeyErrors)
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan

# ------------------------------
# Market Mood (based on confidence & signal)
# ------------------------------
if "confidence" in df.columns and "signal" in df.columns:
    mood_score = (
        df.apply(lambda x: x["confidence"] if str(x["signal"]).upper() == "BUY" else -x["confidence"], axis=1)
        .mean()
        if len(df) > 0 else 0
    )
else:
    mood_score = 0

mood_label = "🟢 Bullish" if mood_score > 10 else ("🟡 Neutral" if -10 <= mood_score <= 10 else "🔴 Bearish")

st.markdown(f"🧭 **Overall Market Mood:** {mood_label} *(score = {round(mood_score/100, 2)})*")
st.success("✅ Live data loaded successfully")

# ------------------------------
# Heat Map by Sector (Average % Change)
# ------------------------------
st.subheader("📊 Sector Heat Map")
if "sector" in df.columns and "pct_change" in df.columns:
    try:
        sector_summary = df.groupby("sector")["pct_change"].mean().sort_values(ascending=False)
        st.bar_chart(sector_summary)
    except Exception:
        st.warning("⚠️ Could not generate sector summary.")
else:
    st.warning("⚠️ Missing required columns for heatmap (`sector` or `pct_change`).")

# ------------------------------
# Data Categories
# ------------------------------
st.subheader("🔥 Top Gainers / 📉 Top Losers / 📊 Most Active / 🧠 All Combined")

try:
    if "pct_change" in df.columns:
        top_gainers = df.sort_values("pct_change", ascending=False).head(10)
        top_losers = df.sort_values("pct_change", ascending=True).head(10)
    else:
        top_gainers = df.head(0)
        top_losers = df.head(0)

    most_active = df.sort_values("volume", ascending=False).head(10) if "volume" in df.columns else df.head(0)
    combined = df.head(25)

    tabs = st.tabs(["🔥 Top Gainers", "📉 Top Losers", "📊 Most Active", "🧠 All Combined"])
    for cat, subdf in zip(["Gainers", "Losers", "Active", "Combined"], [top_gainers, top_losers, most_active, combined]):
        with tabs[["Gainers", "Losers", "Active", "Combined"].index(cat)]:
            if len(subdf) > 0:
                available_cols = [col for col in expected_cols if col in subdf.columns]
                st.dataframe(subdf[available_cols])
            else:
                st.info("No data for this category right now.")

except Exception as e:
    st.warning(f"⚠️ Error generating table: {e}")

# ------------------------------
# News Section (optional integration)
# ------------------------------
st.subheader("📰 Market Buzz")
try:
    buzz = pd.read_csv("data/market_buzz.csv")
    if not buzz.empty:
        st.dataframe(buzz.head(10))
    else:
        st.info("No news available yet.")
except Exception:
    st.info("Market buzz file not found yet.")

# ------------------------------
# Auto Refresh Notice
# ------------------------------
st.markdown(f"⚙️ Auto-refresh every **{REFRESH_INTERVAL} sec**")

