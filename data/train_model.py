"""
train_model.py — Step 3 of the pipeline.

Loads the engineered feature CSV (produced by engineer_features.R), trains
an XGBoost classifier to predict next-day direction (Up/Down), evaluates it
on a held-out test set, and saves:

  - models/xgb_model.json        the trained booster (also loadable from R
                                  via xgboost::xgb.load or xgb.Booster$new)
  - models/feature_columns.json  the exact ordered list of feature columns,
                                  so downstream consumers (the Shiny app,
                                  R Markdown report) build a matching matrix
  - models/eval_log.json         accuracy, confusion matrix, classification
                                  report and feature importances

Note: because this is time-series data, the 80/20 train/test split is
CHRONOLOGICAL (the model trains on the earlier 80% and is evaluated on the
most recent 20%) rather than a random shuffle, to avoid leaking future
information into training.

Usage (run from the project root):
    python data/train_model.py
    python data/train_model.py --features data/features/features.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "features" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"

NON_FEATURE_COLS = {"Date", "Target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost up/down classifier.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature file not found: {args.features} "
            "(run data/engineer_features.R first)"
        )

    print(f"Loading features from {args.features}")
    df = pd.read_csv(args.features, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["Target"].astype(int)

    split_idx = int(len(df) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df["Date"].iloc[split_idx:]

    print(
        f"Chronological split: {len(X_train)} train rows "
        f"({df['Date'].iloc[0].date()} -> {df['Date'].iloc[split_idx - 1].date()}), "
        f"{len(X_test)} test rows "
        f"({dates_test.iloc[0].date()} -> {dates_test.iloc[-1].date()})"
    )

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    print("Training XGBoost classifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["Down", "Up"], output_dict=True)

    # Naive baseline: always predict the majority class in the training set.
    majority_class = int(y_train.mode()[0])
    baseline_accuracy = float((y_test == majority_class).mean())

    print(f"Test accuracy: {accuracy:.4f}  (majority-class baseline: {baseline_accuracy:.4f})")
    print("Confusion matrix [rows=actual Down/Up, cols=predicted Down/Up]:")
    print(cm)

    args.models_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / "xgb_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    feature_cols_path = args.models_dir / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature column order to {feature_cols_path}")

    importances = model.feature_importances_
    feature_importance = sorted(
        [{"feature": c, "importance": float(i)} for c, i in zip(feature_cols, importances)],
        key=lambda d: d["importance"],
        reverse=True,
    )

    eval_log = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_date_range": [str(df["Date"].iloc[0].date()), str(df["Date"].iloc[split_idx - 1].date())],
        "test_date_range": [str(dates_test.iloc[0].date()), str(dates_test.iloc[-1].date())],
        "test_size": args.test_size,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "accuracy": float(accuracy),
        "baseline_accuracy": baseline_accuracy,
        "confusion_matrix": {
            "labels": ["Down", "Up"],
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "feature_importance": feature_importance,
        "feature_columns": feature_cols,
    }

    eval_log_path = args.models_dir / "eval_log.json"
    with open(eval_log_path, "w") as f:
        json.dump(eval_log, f, indent=2)
    print(f"Saved evaluation log to {eval_log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
