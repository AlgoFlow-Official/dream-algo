# src/dashboard/app.py
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from universal_scanner import universal_scan
from news_feed import get_market_news, summarize_news

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Dream Algo v1.6", layout="wide")
st.title("🚀 Dream Algo — Live Universal Market Scanner (AI Enhanced)")
st.caption("v1.6  |  AI-powered signals + real-time market intelligence")

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("🧩 Controls")
refresh = st.sidebar.button("🔁 Run Live Scan Now")
auto_refresh = st.sidebar.checkbox("Auto-Refresh Every 60 s", value=False)
filter_choice = st.sidebar.radio(
    "Filter by View",
    ["All Stocks", "Top Gainers (> 0%)", "High Volume (> 2× avg)"],
    index=0
)
show_rows = st.sidebar.slider("Rows to Display", 10, 100, 30, step=5)

# ---------- LOAD / SCAN DATA ----------
def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "live_signals.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame()
    else:
        st.warning("No data found yet. Click 'Run Live Scan Now' to start.")
        return pd.DataFrame()

if refresh:
    with st.spinner("Running Dream Algo scan (~30 s)…"):
        df = universal_scan(limit=50)
        st.success("✅ AI Scan Complete — Data Updated!")
else:
    df = load_data()

# ---------- FILTER LOGIC ----------
if not df.empty:
    if filter_choice.startswith("Top Gainers"):
        temp = df[df["%_change"] > 0]
        if temp.empty:
            st.info("No positive movers yet — showing all stocks instead.")
        else:
            df = temp
    elif filter_choice.startswith("High Volume"):
        temp = df[df["rel_volume"] > 2]
        if temp.empty:
            st.info("No high-volume tickers yet — showing all stocks instead.")
        else:
            df = temp

# ---------- DISPLAY MAIN TABLE ----------
if df.empty:
    st.info("No data to display yet. Click 'Run Live Scan Now'.")
else:
    st.markdown(f"### 🔥 Top Potential Movers ({filter_choice})")
    display_cols = [
        "ticker", "price", "%_change", "volume", "rel_volume",
        "signal", "confidence", "reason"
    ]
    try:
        st.dataframe(
            df[display_cols]
            .head(show_rows)
            .style.background_gradient(cmap="Greens", subset=["confidence"])
            .set_properties(**{"text-align": "center"})
        )
    except Exception as e:
        st.warning(f"Display warning: {e}")
        st.dataframe(df.head(show_rows))

# ---------- AI MARKET FEED ----------
st.markdown("## 📰 AI Market Feed & News Explainer")
try:
    df_news = get_market_news(limit=20)
    if df_news.empty:
        st.info("No news available right now.")
    else:
        tickers = df["ticker"].astype(str).unique().tolist() if not df.empty else []
        df_summ = summarize_news(df_news, tickers)
        if df_summ.empty:
            st.info("No specific news matched your top movers — showing latest market headlines.")
            st.table(df_news.head(10))
        else:
            st.markdown("### 🔍 Catalysts for Your Top Movers")
            for _, r in df_summ.head(10).iterrows():
                st.markdown(f"**{r['ticker']}** — {r['summary']} ([link]({r['link']}))")
except Exception as e:
    st.warning(f"News feed error: {e}")

# ---------- AUTO-REFRESH LOOP ----------
if auto_refresh:
    countdown = st.empty()
    for sec in range(60, 0, -1):
        countdown.info(f"♻️ Auto-refresh in {sec}s — Uncheck to stop.")
        time.sleep(1)
    st.rerun()

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Dream Algo v1.6  |  Built by Huzzy  |  Live 4 AM – 8 PM EST")
