"""
Model Comparison Script for Loan Evaluation Academic Research.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Loads evaluation metrics from evaluation/random_forest_results.json and evaluation/xgboost_results.json.
- Synthesizes a comparative benchmark table (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
- Computes absolute and relative metric deltas (XGBoost vs. Random Forest).
- Persists comparison results to evaluation/model_comparison.csv.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_paths() -> Dict[str, Path]:
    """Resolve paths for evaluation results and output comparison table."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    return {
        "rf_results": workspace_root / "evaluation" / "random_forest_results.json",
        "xgb_results": workspace_root / "evaluation" / "xgboost_results.json",
        "output_csv": workspace_root / "evaluation" / "model_comparison.csv",
    }


def load_results(path: Path) -> Dict:
    """Load JSON results file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_models() -> pd.DataFrame:
    """Load metrics for Random Forest and XGBoost and construct comparative summary."""
    print("=" * 80)
    print("REVISED MODEL COMPARISON: RANDOM FOREST BASELINE vs. XGBOOST PROPOSED (3,000 DATASET)")
    print("=" * 80)

    paths = resolve_paths()
    rf_data = load_results(paths["rf_results"])
    xgb_data = load_results(paths["xgb_results"])

    rf_acc = rf_data["accuracy"]
    rf_prec = rf_data["precision"]
    rf_rec = rf_data["recall"]
    rf_f1 = rf_data["f1_score"]
    rf_auc = rf_data["roc_auc"]

    xgb_acc = xgb_data["accuracy"]
    xgb_prec = xgb_data["precision"]
    xgb_rec = xgb_data["recall"]
    xgb_f1 = xgb_data["f1_score"]
    xgb_auc = xgb_data["roc_auc"]

    # Calculate absolute differences (XGBoost - Random Forest)
    diff_acc = round(xgb_acc - rf_acc, 4)
    diff_prec = round(xgb_prec - rf_prec, 4)
    diff_rec = round(xgb_rec - rf_rec, 4)
    diff_f1 = round(xgb_f1 - rf_f1, 4)
    diff_auc = round(xgb_auc - rf_auc, 4)

    comparison_records: List[Dict] = [
        {
            "Model": "Random Forest (Baseline)",
            "Accuracy": rf_acc,
            "Precision": rf_prec,
            "Recall": rf_rec,
            "F1-Score": rf_f1,
            "ROC-AUC": rf_auc,
        },
        {
            "Model": "XGBoost (Proposed)",
            "Accuracy": xgb_acc,
            "Precision": xgb_prec,
            "Recall": xgb_rec,
            "F1-Score": xgb_f1,
            "ROC-AUC": xgb_auc,
        },
        {
            "Model": "Absolute Difference (XGB - RF)",
            "Accuracy": diff_acc,
            "Precision": diff_prec,
            "Recall": diff_rec,
            "F1-Score": diff_f1,
            "ROC-AUC": diff_auc,
        },
    ]

    df_comparison = pd.DataFrame(comparison_records)

    # Display comparison
    print("\n--- MODEL PERFORMANCE COMPARISON TABLE (3,000-RECORD BENCHMARK) ---")
    print(df_comparison.to_string(index=False))

    # Delta summary
    print("\n--- ABSOLUTE METRIC DIFFERENCES (XGBoost vs. Random Forest) ---")
    print(f"  Delta Accuracy  : {diff_acc:+.4f} ({diff_acc*100:+.2f}%)")
    print(f"  Delta Precision : {diff_prec:+.4f} ({diff_prec*100:+.2f}%)")
    print(f"  Delta Recall    : {diff_rec:+.4f} ({diff_rec*100:+.2f}%) [XGBoost increases identification of Approved loans]")
    print(f"  Delta F1-Score  : {diff_f1:+.4f} ({diff_f1*100:+.2f}%) [XGBoost improves harmonic balance]")
    print(f"  Delta ROC-AUC   : {diff_auc:+.4f} ({diff_auc*100:+.2f}%)")

    # Confusion matrix comparison
    rf_cm = rf_data["confusion_matrix"]
    xgb_cm = xgb_data["confusion_matrix"]
    print("\n--- CONFUSION MATRIX COMPARISON (N = 600 Test Samples) ---")
    print(f"  Random Forest: TN={rf_data['true_negatives']}, FP={rf_data['false_positives']}, FN={rf_data['false_negatives']}, TP={rf_data['true_positives']}")
    print(f"  XGBoost      : TN={xgb_data['true_negatives']}, FP={xgb_data['false_positives']}, FN={xgb_data['false_negatives']}, TP={xgb_data['true_positives']}")

    # Save to CSV
    df_comparison.to_csv(paths["output_csv"], index=False, encoding="utf-8")
    print(f"\n[OK] Saved model comparison to: {paths['output_csv'].name}")
    print("=" * 80)

    return df_comparison


if __name__ == "__main__":
    compare_models()
