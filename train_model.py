import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib  # for saving and loading models

# Paths
DATASET = "data/dataset_intraday.csv"
MODEL_FILE = "models/random_forest.pkl"

def main():
    # Load dataset
    df = pd.read_csv(DATASET)

    # Features and target
    feature_cols = [
        "open", "close",
        "premarket_gap_pct", "premarket_volume",
        "intraday_return_pct", "volatility_pct",
        "market_cap", "forward_PE", "beta"
    ]
    target_col = "label_pop5"

    # Drop missing rows
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature importance:\n", importance_df)

    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\n[+] Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
