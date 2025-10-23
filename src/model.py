import random

# ---------------------------------
# DREAM ALGO MODEL (placeholder)
# ---------------------------------
# Later, this will use your trained model to make predictions.
# For now, it returns random Buy/Sell signals with random confidence.

def predict_signal(ticker: str):
    """
    Return a (signal, confidence) tuple for the given ticker.
    Example: ('buy', 87)
    """
    try:
        signal = random.choice(["buy", "sell", "hold"])
        confidence = round(random.uniform(60, 99), 2)
        return signal, confidence
    except Exception as e:
        print(f"[ERROR in predict_signal] {e}")
        return "hold", 0
