# src/sentiment_engine.py
"""
Dream Algo — AI Sentiment & Signal Engine (Phase 21 Final)
Generates AI-based buy/sell/watch/neutral signals for each ticker
and computes overall market sentiment (mood + score).
"""

import pandas as pd
import random


# ---------- INDIVIDUAL SIGNAL SCORING ----------
def generate_ai_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns sentiment, signal, confidence, and reasoning to each ticker.
    Heuristic rules simulate an AI layer that later can be replaced by an ML model.
    """
    if df.empty:
        return df

    signals, confidences, reasons = [], [], []

    for _, row in df.iterrows():
        pct = row.get("pct_change", 0)
        if pd.isna(pct):
            pct = 0

        # --- Signal logic ---
        if pct > 10:
            signal = "BUY"
            reason = "Premarket breakout momentum"
            conf = random.randint(88, 98)
        elif pct > 4:
            signal = "WATCH"
            reason = "Uptrend forming with solid volume"
            conf = random.randint(72, 86)
        elif pct < -6:
            signal = "SELL"
            reason = "Bearish reversal or heavy sell-off"
            conf = random.randint(82, 94)
        else:
            signal = "NEUTRAL"
            reason = "Low volatility or sideways movement"
            conf = random.randint(55, 70)

        signals.append(signal)
        confidences.append(conf)
        reasons.append(reason)

    df["signal"] = signals
    df["confidence"] = confidences
    df["reason"] = reasons
    return df


# ---------- MARKET-LEVEL SENTIMENT ----------
def compute_market_mood(df: pd.DataFrame) -> dict:
    """
    Aggregates all ticker signals to produce an overall market mood and numeric score.
    Score ranges roughly between -1.0 (bearish) and +1.0 (bullish).
    """
    if df.empty or "signal" not in df.columns:
        return {"mood": "Neutral", "score": 0.0}

    # Assign sentiment weights
    weight_map = {"BUY": 1, "WATCH": 0.5, "NEUTRAL": 0, "SELL": -1}
    avg_score = df["signal"].map(weight_map).mean()

    if avg_score > 0.25:
        mood = "Bullish"
    elif avg_score < -0.25:
        mood = "Bearish"
    else:
        mood = "Neutral"

    return {"mood": mood, "score": round(avg_score, 2)}
