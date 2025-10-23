
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.base import clone
from xgboost import XGBClassifier

# -------- Settings ----------
DATA_PATH = "data/dataset_intraday.csv"

# Default labeling params (can be overridden by CLI args)
LABEL_MODE = "direction"   # "direction" (next bar up/down) or "magnitude"
HORIZON = 1                # how many bars ahead (only used in magnitude mode)
THRESHOLD = 0.01           # +1% move threshold (only used in magnitude mode)

# Threshold tuning settings
TUNE_THRESHOLD = True
THRESH_SCAN = np.linspace(0.30, 0.70, 21)  # range of thresholds to test

# XGB hyperparam grid
PARAM_GRID = {
    "n_estimators": [200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "reg_lambda": [1, 5, 10],  # L2 regularization
    "reg_alpha": [0, 0.5],     # L1 regularization
}
N_SPLITS = 5
# ----------------------------

print = lambda *a, **k: __builtins__.print(*a, **{**k, "flush": True})


# ---------- Data Helpers ----------
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.dropna()
    return df


def build_xy(df, mode="direction", horizon=1, threshold=0.01):
    feats = ["gap_pct","volatility","volume_norm","rvol_10d",
             "vwap_dev","momentum_5","rsi_14","macd","macd_signal","macd_hist"]
    X = df[feats].copy()

    if mode == "direction":
        future = df["Close"].shift(-1)
        y = (future > df["Close"]).astype(int)
        cutoff = 1
    else:
        future = df["Close"].shift(-horizon)
        ret = (future - df["Close"]) / df["Close"]
        y = (ret >= threshold).astype(int)
        cutoff = horizon

    if cutoff > 0:
        X = X.iloc[:-cutoff]
        y = y.iloc[:-cutoff]

    return X, y, feats


def class_weight(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0, pos, neg
    return max(neg/pos, 1.0), pos, neg


# ---------- Evaluation Helpers ----------
def tune_decision_threshold(estimator, X_fit, y_fit, X_val, y_val, scan):
    model = clone(estimator)
    model.fit(X_fit, y_fit)
    p = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in scan:
        y_hat = (p >= t).astype(int)
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1, model


def evaluate(name, model, X, y, threshold=0.5):
    p = model.predict_proba(X)[:, 1]
    y_hat = (p >= threshold).astype(int)
    acc = accuracy_score(y, y_hat)
    prec = precision_score(y, y_hat, zero_division=0)
    rec = recall_score(y, y_hat, zero_division=0)
    f1 = f1_score(y, y_hat, zero_division=0)
    print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return y_hat, p


def plot_feature_importance(model, features, out_path):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] Model has no feature_importances_")
        return
    order = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(features)[order], np.array(importances)[order])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance (Regularized)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[INFO] Saved feature importance -> {out_path}")


def save_predictions(X, y, probs, y_hat_default, y_hat_tuned, out_path):
    df_out = pd.DataFrame({
        "Datetime": X.index if hasattr(X, "index") else np.arange(len(y)),
        "True_Label": y.values,
        "Prob": probs,
        "Pred_Default_0.5": y_hat_default,
        "Pred_Tuned": y_hat_tuned
    })
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions -> {out_path}")


# ---------- Main ----------
if __name__ == "__main__":
    # CLI args
    args = sys.argv[1:]
    if len(args) >= 1:
        LABEL_MODE = args[0]
    if len(args) >= 2:
        HORIZON = int(args[1])
    if len(args) >= 3:
        THRESHOLD = float(args[2])

    print(f"[INFO] CLI args => mode={LABEL_MODE}, horizon={HORIZON}, threshold={THRESHOLD}")

    # Load dataset
    df = load_dataset()
    print(f"[INFO] Loaded dataset with shape {df.shape}")

    if df.empty:
        print("[FATAL] Dataset is empty. Run build_dataset.py first.")
        sys.exit(1)

    # Build labels/features
    X_all, y_all, features = build_xy(df, mode=LABEL_MODE, horizon=HORIZON, threshold=THRESHOLD)
    print(f"[INFO] Label mode: {LABEL_MODE} | horizon={HORIZON} | threshold={THRESHOLD}")
    print(f"[INFO] Samples after alignment: {X_all.shape[0]}")

    # Train/test split
    split_idx = int(len(X_all)*0.8)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    # Handle class imbalance
    spw, pos, neg = class_weight(y_train)
    print(f"[INFO] Train class balance => pos={pos}, neg={neg}, scale_pos_weight={spw:.2f}")

    # XGBoost with CV
    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        scale_pos_weight=spw
    )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    grid = GridSearchCV(base_xgb, PARAM_GRID, cv=tscv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\n[CV RESULTS]")
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    # Threshold tuning
    best_thresh = 0.5
    tuned_note = ""
    if TUNE_THRESHOLD:
        cut = int(len(X_train)*0.9)
        X_fit, y_fit = X_train.iloc[:cut], y_train.iloc[:cut]
        X_val, y_val = X_train.iloc[cut:], y_train.iloc[cut:]
        best_thresh, best_f1, _ = tune_decision_threshold(
            grid.best_estimator_, X_fit, y_fit, X_val, y_val, THRESH_SCAN
        )
        tuned_note = f"(tuned threshold={best_thresh:.2f}, val F1={best_f1:.3f})"
    print(f"[INFO] Using decision threshold={best_thresh:.2f} {tuned_note}")

    # Evaluate on test set
    print("\n[TEST SET EVALUATION]")
    y_hat_default, probs = evaluate("XGBoost @0.50", grid.best_estimator_, X_test, y_test, threshold=0.50)
    y_hat_tuned, _ = evaluate("XGBoost @tuned", grid.best_estimator_, X_test, y_test, threshold=best_thresh)

    # Save predictions CSV
    save_predictions(X_test, y_test, probs, y_hat_default, y_hat_tuned, "data/test_predictions_xgb.csv")

    # Feature importance plot
    plot_feature_importance(grid.best_estimator_, features, "data/feature_importance_xgb_regularized.png")

import joblib, os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# --- Save best model ---
try:
    joblib.dump(grid.best_estimator_, "models/xgb_model.pkl")
    print("[INFO] Saved trained model -> models/xgb_model.pkl")
except Exception as e:
    print(f"[WARN] Could not save model: {e}")

# --- Save tuned threshold if available ---
if "best_thresh" in locals():
    try:
        joblib.dump(best_thresh, "models/tuned_threshold.pkl")
        print(f"[INFO] Saved tuned threshold -> models/tuned_threshold.pkl (value={best_thresh})")
    except Exception as e:
        print(f"[WARN] Could not save threshold: {e}")
else:
    print("[INFO] No tuned threshold found to save.")


