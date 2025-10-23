import os
import pandas as pd
import joblib
from tabulate import tabulate

# Paths
MODEL_FILE = "models/random_forest.pkl"
DATASET = "data/dataset_intraday.csv"  # use dataset for now

def main():
    # Load trained model
    model = joblib.load(MODEL_FILE)
    print(f"[+] Loaded model from {MODEL_FILE}")

    # Load dataset
    df = pd.read_csv(DATASET)

    # Features
    feature_cols = [
        "open", "close",
        "premarket_gap_pct", "premarket_volume",
        "intraday_return_pct", "volatility_pct",
        "market_cap", "forward_PE", "beta"
    ]
    df = df.dropna(subset=feature_cols)

    # Predictions
    X = df[feature_cols]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Attach results
    df["prediction"] = preds
    df["pop_probability"] = probs

    # Show top candidates
    results = df[["ticker", "premarket_gap_pct", "premarket_volume", "prediction", "pop_probability"]]
    results = results.sort_values(by="pop_probability", ascending=False)

    print("\nPredictions (from dataset):\n")
    print(tabulate(results.head(15), headers="keys", tablefmt="psql"))

    # Save to CSV
    os.makedirs("logs", exist_ok=True)
    out_file = "logs/predictions_from_dataset.csv"
    results.to_csv(out_file, index=False)
    print(f"\n[+] Predictions saved to {out_file}")

if __name__ == "__main__":
    main()
