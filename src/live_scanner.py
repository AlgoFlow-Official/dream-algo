import time
import datetime
from scanner import get_top_movers
from model import predict_signal
from google_sheets import update_google_sheet
from ai_feed import fetch_news, summarize_news

# -----------------------------
# DREAM ALGO LIVE SCANNER + AI FEED
# -----------------------------

START_HOUR = 4      # 4 AM
END_HOUR = 20       # 8 PM
REFRESH_INTERVAL = 300  # 5 minutes

def within_market_hours():
    """Return True if current time between 4 AM – 8 PM."""
    hour = datetime.datetime.now().hour
    return START_HOUR <= hour <= END_HOUR


def run_live():
    print("🚀 Dream Algo Live Scanner + AI Feed Running (4 AM – 8 PM)…")

    while True:
        if within_market_hours():
            try:
                # 1️⃣ Get top movers
                movers = get_top_movers(limit=20)
                print(f"📊 Fetched {len(movers)} tickers")

                # 2️⃣ Predict signals + confidence
                for m in movers:
                    signal, conf = predict_signal(m["ticker"])
                    m["signal"], m["confidence"] = signal, conf

                    # 3️⃣ Fetch latest news and AI summary
                    articles = fetch_news(m["ticker"])
                    m["ai_summary"] = summarize_news(m["ticker"], articles)

                # 4️⃣ Update Google Sheet
                update_google_sheet(movers)
                print(f"[{datetime.datetime.now()}] ✅ Sheet updated ({len(movers)} stocks)")

            except Exception as e:
                print(f"[ERROR Live Scanner] {e}")

        else:
            print("🌙 Market closed – sleeping 30 min…")
            time.sleep(1800)
            continue

        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    run_live()
