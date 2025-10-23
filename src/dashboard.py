# src/dashboard.py
"""
Dream Algo — V1.7 Universal Dashboard (Phase 22 Final)
Live AI market scanner + sentiment heat map + global news feed.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
import os
import plotly.express as px

from sentiment_engine import compute_market_mood
from news_engine import get_market_news

# -------- CONFIG --------
MARKET_TZ = pytz.timezone("US/Eastern")
DATA_PATH = "data/live_movers.csv"
REFRESH_SEC = 60

# -------- PAGE SETTINGS --------
st.set_page_config(page_title="Dream Algo — Global Dashboard", layout="wide")
st.title("💎 Dream Algo V1.7 — Universal AI Market Dashboard")
st.caption("Real-time market analytics • 4 AM – 8 PM EST")

# -------- LOAD DATA --------
@st.cache_data(ttl=REFRESH_SEC)
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame()

df = load_data()
now_est = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M EST")

# -------- SIDEBAR STATUS --------
st.sidebar.markdown(f"**🕒 Last Updated:** {now_est}")
st.sidebar.markdown(f"**📁 Rows Loaded:** {len(df)}")
st.sidebar.markdown(f"**⚙️ Auto-Refresh:** {REFRESH_SEC} sec")

# -------- MARKET MOOD GAUGE --------
if not df.empty:
    mood = compute_market_mood(df)
    color = "🟢" if mood["mood"] == "Bullish" else "🔴" if mood["mood"] == "Bearish" else "🟡"
    st.markdown(f"### 🧭 Overall Market Mood: {color} **{mood['mood']}** (score = {mood['score']})")

# -------- MAIN DATA --------
if df.empty:
    st.warning("No live data yet — wait for the scanner to update.")
else:
    st.success("✅ Live data loaded successfully")

    # Simulated sectors if not provided
    if "sector" not in df.columns:
        sectors = [
            "Technology", "Healthcare", "Finance", "Energy",
            "Industrials", "Consumer", "Utilities", "Materials"
        ]
        df["sector"] = [sectors[i % len(sectors)] for i in range(len(df))]

    # -------- SECTOR HEAT MAP --------
    sector_perf = (
        df.groupby("sector")["pct_change"]
        .mean()
        .reset_index()
        .sort_values("pct_change", ascending=False)
    )
    fig = px.treemap(
        sector_perf,
        path=["sector"],
        values="pct_change",
        color="pct_change",
        color_continuous_scale="RdYlGn",
        title="📊 Sector Performance Heat Map (Avg % Change)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------- TABS SECTION --------
    tabs = st.tabs([
        "🔥 Top Gainers",
        "📉 Top Losers",
        "📊 Most Active",
        "🧠 All Combined",
        "📰 Market Buzz"
    ])

    # ---- Stock Tables ----
    for label, cat in zip(tabs[:3], ["Top Gainers", "Top Losers", "Most Active"]):
        with label:
            sub = df[df["category"] == cat]
            if not sub.empty:
                st.dataframe(
                    sub[[
                        "ticker", "name", "price", "pct_change",
                        "volume", "signal", "confidence", "reason", "sector"
                    ]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No data for this category right now.")

    # ---- Combined ----
    with tabs[3]:
        st.dataframe(
            df[[
                "ticker", "name", "price", "pct_change",
                "volume", "signal", "confidence", "reason", "sector", "category"
            ]],
            width="stretch",
            hide_index=True,
        )

    # ---- Market Buzz News Feed ----
    with tabs[4]:
        st.subheader("📰 Top Market Headlines — AI Classified")
        try:
            news_df = get_market_news(30)
            if news_df.empty:
                st.warning("No news fetched yet.")
            else:
                st.dataframe(
                    news_df[["source", "headline", "category", "published"]],
                    hide_index=True,
                    width="stretch",
                )
        except Exception as e:
            st.error(f"⚠️ News fetch error: {e}")

# -------- AUTO-REFRESH (Meta) --------
st.toast(f"🔄 Auto-refresh every {REFRESH_SEC} sec…", icon="⏱️")
st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SEC}'>",
    unsafe_allow_html=True,
)
