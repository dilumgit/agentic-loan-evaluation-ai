"""
Independent Validation & Evaluation Script for Random Forest Baseline Model.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Loads the persisted Random Forest baseline model artifact from models/baseline/random_forest.joblib.
- Loads the current test dataset from data/processed/X_test.csv and y_test.csv (600 samples, 55 features).
- Re-runs evaluation to independently verify performance metrics and ensure complete reproducibility.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_paths() -> Dict[str, Path]:
    """Resolve paths for model artifact, test datasets, and evaluation reports."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    return {
        "model": workspace_root / "models" / "baseline" / "random_forest.joblib",
        "X_test": workspace_root / "data" / "processed" / "X_test.csv",
        "y_test": workspace_root / "data" / "processed" / "y_test.csv",
        "results_json": workspace_root / "evaluation" / "random_forest_results.json",
    }


def evaluate_saved_baseline() -> None:
    """Load the saved model and evaluate on the test split."""
    print("=" * 80)
    print("INDEPENDENT EVALUATION OF SAVED RANDOM FOREST BASELINE MODEL")
    print("=" * 80)

    paths = resolve_paths()

    if not paths["model"].exists():
        raise FileNotFoundError(f"Saved baseline model not found at: {paths['model']}")
    if not paths["X_test"].exists() or not paths["y_test"].exists():
        raise FileNotFoundError("Processed test datasets not found in data/processed/")

    # 1. Load Model & Data
    print(f"Loading Model from : {paths['model']}")
    rf_model = joblib.load(paths["model"])

    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]
    print(f"Test Dataset Shape : {X_test.shape[0]} samples, {X_test.shape[1]} features (Expected: 600, 55)")

    if X_test.shape != (600, 55):
        raise ValueError(f"Expected test shape (600, 55), found: {X_test.shape}")

    # 2. Predictions & Probabilities
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    # 3. Compute Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 4. Display Results
    print("\n--- INDEPENDENT REPRODUCIBILITY VALIDATION METRICS ---")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"  [TN={tn}, FP={fp}]")
    print(f"  [FN={fn}, TP={tp}]")
    print(f"  True Positives  (TP) : {tp}")
    print(f"  True Negatives  (TN) : {tn}")
    print(f"  False Positives (FP) : {fp}")
    print(f"  False Negatives (FN) : {fn}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

    # Verify against saved JSON results if present
    if paths["results_json"].exists():
        with open(paths["results_json"], "r", encoding="utf-8") as f:
            saved_results = json.load(f)
        assert round(acc, 4) == saved_results["accuracy"], "Accuracy mismatch with saved results!"
        assert round(f1, 4) == saved_results["f1_score"], "F1 mismatch with saved results!"
        assert int(tp) == saved_results["true_positives"], "TP mismatch with saved results!"
        assert int(tn) == saved_results["true_negatives"], "TN mismatch with saved results!"
        print("[OK] Independent validation matches saved results JSON with 100% precision.")

    print("\n" + "=" * 80)
    print("[SUCCESS] BASELINE MODEL EVALUATION COMPLETED & VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_saved_baseline()
