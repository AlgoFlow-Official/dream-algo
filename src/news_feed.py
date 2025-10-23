# src/news_feed.py
"""
Dream Algo v1.6 — News & Market Explainer (Enhanced)
Primary: Yahoo Finance RSS
Backup: MarketWatch RSS
"""

import requests, re
import pandas as pd
from datetime import datetime

def fetch_rss(url):
    """Fetch headlines from any RSS feed."""
    try:
        xml = requests.get(url, timeout=10).text
        items = re.findall(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>", xml, re.S)
        headlines = [
            {"title": re.sub("<.*?>", "", t).strip(), "link": l.strip()}
            for t, l in items
        ]
        return pd.DataFrame(headlines)
    except Exception as e:
        print("⚠️ RSS fetch failed:", e)
        return pd.DataFrame()

def get_market_news(limit: int = 20):
    """Get top market headlines from Yahoo or fallback MarketWatch."""
    urls = [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]
    df = pd.DataFrame()
    for url in urls:
        df = fetch_rss(url)
        if not df.empty:
            break
    return df.head(limit)

def summarize_news(df_news, tickers):
    """Generate short AI-style summaries for tickers appearing in news."""
    summaries = []
    if df_news.empty:
        return pd.DataFrame()

    for _, row in df_news.iterrows():
        title = row["title"]
        link = row["link"]
        upper_title = title.upper()
        match = [tk for tk in tickers if tk in upper_title]
        if match:
            for tk in match:
                if "EARN" in upper_title:
                    reason = "Earnings beat"
                elif "UPGRADE" in upper_title:
                    reason = "Analyst upgrade"
                elif "PARTNER" in upper_title:
                    reason = "Partnership news"
                elif "MERGER" in upper_title or "ACQUI" in upper_title:
                    reason = "Merger/Acquisition"
                else:
                    reason = "Market momentum"
                summaries.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "ticker": tk,
                    "headline": title,
                    "summary": f"{reason} — {title[:90]}…",
                    "link": link
                })
    return pd.DataFrame(summaries)
