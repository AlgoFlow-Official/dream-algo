import requests
from datetime import datetime, timedelta

def fetch_market_news():
    """
    Fetches market news with fallback and lightweight AI summarizer.
    Returns a list of dicts with title, source, and summary.
    """
    try:
        # Example: Using Yahoo Finance RSS feed (no API key needed)
        url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200 and "<item>" in resp.text:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            items = root.findall(".//item")

            news_list = []
            for item in items[:10]:
                title = item.find("title").text if item.find("title") is not None else "No title"
                link = item.find("link").text if item.find("link") is not None else "#"
                source = "Yahoo Finance"
                # AI-like summary (basic keyword reasoning)
                summary = ai_summarizer(title)
                news_list.append({
                    "title": title.strip(),
                    "source": source,
                    "link": link.strip(),
                    "summary": summary
                })
            return news_list

        else:
            print("[WARN] RSS request returned no items.")
            return fallback_news()

    except Exception as e:
        print(f"[ERROR] News fetch failed: {e}")
        return fallback_news()


def ai_summarizer(title: str) -> str:
    """
    Lightweight text generator to mimic an AI insight based on title content.
    """
    t = title.lower()
    if "rally" in t or "gain" in t or "rise" in t:
        return "AI Insight: Market sentiment appears bullish amid recent gains."
    elif "fall" in t or "drop" in t or "loss" in t:
        return "AI Insight: Cautious mood as investors react to potential selloffs."
    elif "fed" in t or "inflation" in t:
        return "AI Insight: Focus shifts to macroeconomic indicators and rate expectations."
    elif "earnings" in t:
        return "AI Insight: Corporate earnings remain a key driver for premarket moves."
    else:
        return "AI Insight: Market activity remains high ahead of the open."


def fallback_news():
    """
    Simple placeholder fallback data if news API fails.
    """
    now = datetime.now().strftime("%Y-%m-%d")
    return [
        {
            "title": f"Markets hold steady amid mixed sentiment ({now})",
            "source": "Dream Algo Feed",
            "summary": "AI Insight: Investors await catalysts in early premarket trading.",
        },
        {
            "title": "Tech sector volatility remains in focus",
            "source": "Dream Algo Feed",
            "summary": "AI Insight: Traders eye potential reversals in high-volume tickers.",
        },
    ]
