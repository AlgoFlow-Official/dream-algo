import requests
from datetime import datetime

# -------------------------------------
# DREAM ALGO AI FEED
# -------------------------------------
# Pulls latest news for a ticker and summarizes it
# -------------------------------------

NEWS_API_KEY = "87b5de5ca00045cca595c53514eb7820"  # Replace this later

def fetch_news(ticker):
    """
    Fetches top 3 recent news articles related to the given stock ticker.
    Returns a list of {title, source, publishedAt}.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "articles" not in data:
            return []

        articles = data["articles"][:3]
        return [
            {
                "title": a.get("title", "No title"),
                "source": a.get("source", {}).get("name", "Unknown"),
                "publishedAt": a.get("publishedAt", "Unknown")
            }
            for a in articles
        ]

    except Exception as e:
        print(f"[ERROR] Failed to fetch news for {ticker}: {e}")
        return []

def summarize_news(ticker, articles):
    """
    Converts a list of headlines into a short summary.
    Example:
    'TSLA: Elon Musk comments on AI | Strong deliveries | Market optimism'
    """
    if not articles:
        return f"{ticker}: No major headlines right now."

    headlines = " | ".join(a["title"] for a in articles)
    summary = f"{ticker}: {headlines}"
    return summary

if __name__ == "__main__":
    sample = fetch_news("AAPL")
    print(summarize_news("AAPL", sample))
