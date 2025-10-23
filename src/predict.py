import pandas as pd
import joblib
import os

# ---------- Settings ----------
MODEL_PATH = "models/xgb_model.pkl"
DATASET_PATH = "data/dataset_intraday.csv"
OUT_PATH = "data/test_predictions_xgb.csv"
# -------------------------------

# Must match training features exactly
FEATURE_COLS = [
    "gap_pct", "volatility", "volume_norm", "rvol_10d",
    "vwap_dev", "momentum_5", "rsi_14",
    "macd", "macd_signal", "macd_hist"
]

def load_data():
    """Load dataset with Datetime + features."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["Datetime"])
    return df

def predict():
    # Load dataset
    df = load_data()

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Extract features (match training)
    X = df[FEATURE_COLS]

    # Predictions
    probs = model.predict_proba(X)[:, 1]
    preds_default = (probs >= 0.5).astype(int)

    # Load tuned threshold if available
    try:
        tuned_thresh = joblib.load("models/tuned_threshold.pkl")
        print(f"[INFO] Using tuned threshold={tuned_thresh:.2f}")
    except:
        tuned_thresh = 0.5
        print("[WARN] No tuned threshold found, using 0.50")
    preds_tuned = (probs >= tuned_thresh).astype(int)

    # Save results with real Datetime
    out = pd.DataFrame({
        "Datetime": df["Datetime"],
        "True_Label": df.get("True_Label", pd.Series([None]*len(df))),
        "Prob": probs,
        "Pred_Default_0.5": preds_default,
        "Pred_Tuned": preds_tuned
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Saved predictions -> {OUT_PATH} with {len(out)} rows")

if __name__ == "__main__":
    predict()
