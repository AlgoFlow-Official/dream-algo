import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Always flush prints immediately
print = lambda *args, **kwargs: __builtins__.print(*args, **{**kwargs, "flush": True})

DATA_PATH = "data/dataset_intraday.csv"

# ---------- Load Dataset ----------
def load_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    df = df.dropna()
    return df

# ---------- Build Features & Labels ----------
def build_xy(df):
    features = [
        "gap_pct", "volatility", "volume_norm", "rvol_10d",
        "vwap_dev", "momentum_5", "rsi_14", "macd", "macd_signal", "macd_hist"
    ]
    X = df[features]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = y.iloc[:-1]
    X = X.iloc[:-1]
    return X, y, features

# ---------- Train/Test Split ----------
def train_test_split_time(df, split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

# ---------- Learning Curve Plot ----------
def plot_learning_curve(estimator, X, y, title, filename):
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=tscv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training Accuracy", marker="o")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker="o")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[INFO] Saved learning curve plot -> {filename}")

# ---------- Hyperparameter Tuning + Evaluation ----------
def tune_and_evaluate(X_train, y_train, X_test, y_test, features):
    tscv = TimeSeriesSplit(n_splits=5)

    # Logistic Regression Pipeline
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    logreg_params = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
    }
    logreg_grid = GridSearchCV(
        logreg_pipe, logreg_params, cv=tscv, scoring="accuracy", n_jobs=-1
    )
    logreg_grid.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    }
    rf_grid = GridSearchCV(
        rf, rf_params, cv=tscv, scoring="accuracy", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    # Print best CV results
    print("\n[CV RESULTS]")
    print("Logistic Regression best params:", logreg_grid.best_params_)
    print("Logistic Regression best score:", logreg_grid.best_score_)
    print("RandomForest best params:", rf_grid.best_params_)
    print("RandomForest best score:", rf_grid.best_score_)

    # Final test set evaluation
    print("\n[TEST SET EVALUATION]")
    models = {
        "LogisticRegression": logreg_grid.best_estimator_,
        "RandomForest": rf_grid.best_estimator_
    }
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # ---------- Feature Importance Plot ----------
    best_rf = rf_grid.best_estimator_
    importances = best_rf.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("RandomForest Feature Importance")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    print("[INFO] Saved feature importance plot -> data/feature_importance.png")

    # ---------- Learning Curves ----------
    plot_learning_curve(logreg_grid.best_estimator_, X_train, y_train,
                        "Learning Curve (Logistic Regression)", "data/learning_curve_logreg.png")
    plot_learning_curve(rf_grid.best_estimator_, X_train, y_train,
                        "Learning Curve (Random Forest)", "data/learning_curve_rf.png")

# ---------- Main ----------
if __name__ == "__main__":
    df = load_dataset()
    print(f"[INFO] Loaded dataset with shape {df.shape}")

    if df.empty:
        print("[FATAL] Dataset is empty. Run build_dataset.py first.")
        sys.exit(1)

    train_df, test_df = train_test_split_time(df)
    X_train, y_train, features = build_xy(train_df)
    X_test, y_test, _ = build_xy(test_df)

    tune_and_evaluate(X_train, y_train, X_test, y_test, features)
