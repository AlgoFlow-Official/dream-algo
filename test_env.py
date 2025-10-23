from dotenv import load_dotenv
import os

load_dotenv()

print("Finnhub key loaded:", bool(os.getenv("FINNHUB_KEY")))
print("Alpaca key loaded:", bool(os.getenv("ALPACA_KEY")))

# optional: show first few characters (just to confirm without exposing the key)
key = os.getenv("FINNHUB_KEY")
if key:
    print("Finnhub key begins with:", key[:5], "...")
