import yfinance as yf
from datetime import datetime
import math

def get_top_movers(limit=20):
    try:
        universe = [
            "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META", "NFLX",
            "AMD", "PLTR", "BABA", "SPY", "QQQ", "NIO", "SOFI", "COIN",
            "SMCI", "F", "INTC", "CRWD", "NVDS", "SHOP", "SNAP", "ROKU"
        ]

        # Fetch 5-min interval for today
        data = yf.download(
            tickers=universe,
            period="1d",
            interval="5m",
            progress=False,
            group_by="ticker",
            threads=True
        )

        movers = []

        for ticker in universe:
            try:
                df = data[ticker].dropna()
                if df.empty:
                    continue

                last = df.iloc[-1]
                open_price = df["Open"].iloc[0]
                price = round(last["Close"], 2)
                change_pct = round(((price - open_price) / open_price) * 100, 2)
                total_volume = int(df["Volume"].sum())

                if (
                    not math.isnan(price)
                    and not math.isnan(change_pct)
                    and total_volume > 50000
                ):
                    movers.append({
                        "ticker": ticker,
                        "price": price,
                        "change_pct": change_pct,
                        "volume": total_volume
                    })

            except Exception as inner_err:
                print(f"[WARN] Skipping {ticker}: {inner_err}")
                continue

        movers = sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:limit]
        print(f"[{datetime.now()}] 🔍 Top {len(movers)} movers fetched.")
        return movers

    except Exception as e:
        print(f"[ERROR in get_top_movers] {e}")
        return []
