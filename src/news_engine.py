# src/news_engine.py
"""
Dream Algo — Global News AI Feed (Phase 22 Final)
Fetches real-time market headlines from Yahoo Finance + Finviz RSS
and classifies catalysts (earnings beat, FDA approval, analyst upgrade, etc.).
"""

import requests, feedparser, re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

MARKET_TZ = pytz.timezone("US/Eastern")

# ---------- FETCH YAHOO NEWS ----------
def fetch_yahoo_news(limit: int = 25) -> pd.DataFrame:
    url = "https://finance.yahoo.com/rss/topstories"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:limit]:
        items.append({
            "source": "Yahoo Finance",
            "headline": e.title,
            "link": e.link,
            "published": e.published
        })
    return pd.DataFrame(items)


# ---------- FETCH FINVIZ NEWS ----------
def fetch_finviz_news(limit: int = 25) -> pd.DataFrame:
    url = "https://finviz.com/news.ashx"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("table.news tr")
    data = []
    for r in rows[:limit]:
        a = r.find("a")
        if a and a.text:
            data.append({
                "source": "Finviz",
                "headline": a.text.strip(),
                "link": a["href"],
                "published": datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M EST")
            })
    return pd.DataFrame(data)


# ---------- SIMPLE AI-STYLE CATALYST TAGGING ----------
def classify_headline(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["earnings", "q", "revenue", "profit"]):
        return "Earnings Beat/Miss"
    if any(k in t for k in ["upgrade", "downgrade", "rating"]):
        return "Analyst Action"
    if any(k in t for k in ["merger", "acquisition", "buyout"]):
        return "M&A Activity"
    if any(k in t for k in ["approval", "fda", "trial"]):
        return "FDA/Drug News"
    if any(k in t for k in ["partnership", "collaboration"]):
        return "Partnership Deal"
    if any(k in t for k in ["sec", "lawsuit", "investigation"]):
        return "Regulatory/Legal"
    return "General Market"


# ---------- MASTER FETCH FUNCTION ----------
def get_market_news(limit_total: int = 50) -> pd.DataFrame:
    yahoo_df = fetch_yahoo_news(limit_total // 2)
    finviz_df = fetch_finviz_news(limit_total // 2)
    df = pd.concat([yahoo_df, finviz_df], ignore_index=True)
    df["category"] = df["headline"].apply(classify_headline)
    df["timestamp"] = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M EST")
    return df


if __name__ == "__main__":
    news = get_market_news()
    print(news.head(10))
