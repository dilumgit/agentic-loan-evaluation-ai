"""
Final XGBoost Proposed Model Training, Cross-Validation, and Comparative Evaluation Pipeline.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Dataset:
- Dataset v2: loan_evaluation_dataset_3000_v2.csv (500 Empirical + 2,500 Gaussian Copula synthetic records)
- Preprocessed: data/processed/ (X_train: 2,400 x 55, X_test: 600 x 55)

Module Purpose:
- Trains the final proposed XGBoost Classifier on X_train (n_estimators=200, max_depth=4, lr=0.05, scale_pos_weight=1.1486).
- Evaluates on the holdout test set (N=600).
- Performs 5-fold Stratified Cross-Validation strictly on training data (N=2,400).
- Extracts native Gain feature importances mapped with feature_names.json.
- Generates ROC and Precision-Recall curves.
- Compares Final Random Forest Baseline vs. Final XGBoost Proposed Model.
- Persists all artifacts to models/xgboost/ and evaluation/.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


OBJECTIVE = "binary:logistic"
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
RANDOM_STATE = 42
EVAL_METRIC = "logloss"
N_JOBS = -1
DATASET_NAME = "loan_evaluation_dataset_3000_v2.csv (Gaussian Copula Augmented)"


def resolve_paths() -> Dict[str, Path]:
    """Resolve directory and file paths."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent

    data_dir = workspace_root / "data" / "processed"
    models_dir = workspace_root / "models" / "xgboost"
    eval_dir = workspace_root / "evaluation"

    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "X_train": data_dir / "X_train.csv",
        "X_test": data_dir / "X_test.csv",
        "y_train": data_dir / "y_train.csv",
        "y_test": data_dir / "y_test.csv",
        "feature_names": data_dir / "feature_names.json",
        "rf_results": eval_dir / "final_random_forest_results.json",
        "model_output": models_dir / "final_xgboost.joblib",
        "cm_output": eval_dir / "final_xgboost_confusion_matrix.csv",
        "cr_output": eval_dir / "final_xgboost_classification_report.json",
        "pred_output": eval_dir / "final_xgboost_predictions.csv",
        "cv_output": eval_dir / "final_xgboost_cross_validation.csv",
        "imp_output": eval_dir / "final_xgboost_feature_importance.csv",
        "roc_output": eval_dir / "final_xgboost_roc_curve.png",
        "pr_output": eval_dir / "final_xgboost_precision_recall_curve.png",
        "results_json": eval_dir / "final_xgboost_results.json",
        "comparison_csv": eval_dir / "final_model_comparison.csv",
    }


def run_pipeline():
    print("=" * 85)
    print("REVISED STEP 10: FINAL XGBOOST PROPOSED MODEL (DATASET v2 — GAUSSIAN COPULA)")
    print("=" * 85)

    paths = resolve_paths()

    # 1. Load Data
    X_train = pd.read_csv(paths["X_train"])
    X_test = pd.read_csv(paths["X_test"])
    y_train = pd.read_csv(paths["y_train"]).iloc[:, 0]
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]

    with open(paths["feature_names"], "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    tr_neg = int((y_train == 0).sum())
    tr_pos = int((y_train == 1).sum())
    scale_pos_weight = round(tr_neg / tr_pos, 4)

    print(f"Loaded Training Matrix : {X_train.shape} (Approved={tr_pos}, Rejected={tr_neg})")
    print(f"Loaded Testing Matrix  : {X_test.shape} (Approved={int((y_test==1).sum())}, Rejected={int((y_test==0).sum())})")
    print(f"Calculated scale_pos_weight = {scale_pos_weight} (Negative/Positive = {tr_neg}/{tr_pos})")

    # 2. Train Model
    print(f"\n[1] Training Final XGBoost Proposed Model (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, lr={LEARNING_RATE}, scale_pos_weight={scale_pos_weight})...")
    xgb_model = XGBClassifier(
        objective=OBJECTIVE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
        n_jobs=N_JOBS,
    )
    xgb_model.fit(X_train, y_train)
    print("[OK] Model training complete.")

    # Save Model Artifact (Task 2)
    joblib.dump(xgb_model, paths["model_output"])
    print(f"[OK] Saved model binary to: {paths['model_output'].name}")

    # 3. Holdout Evaluation (Task 3)
    print("\n[2] Evaluating model on untouched holdout test set (N=600)...")
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    avg_prec = float(average_precision_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 50)
    print("FINAL XGBOOST TEST SET PERFORMANCE METRICS")
    print("-" * 50)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Avg Prec  : {avg_prec:.4f}")
    print(f"Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 4. Save Confusion Matrix (Task 4)
    df_cm = pd.DataFrame(
        cm,
        index=["Actual_Rejected (0)", "Actual_Approved (1)"],
        columns=["Predicted_Rejected (0)", "Predicted_Approved (1)"],
    )
    df_cm.to_csv(paths["cm_output"], encoding="utf-8")
    print(f"[OK] Saved Confusion Matrix to: {paths['cm_output'].name}")

    # 5. Save Classification Report (Task 5)
    report_dict = classification_report(
        y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], output_dict=True
    )
    with open(paths["cr_output"], "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[OK] Saved Classification Report to: {paths['cr_output'].name}")

    # 6. Save Predictions (Task 6)
    df_preds = pd.DataFrame({
        "Actual Label": y_test.values,
        "Predicted Label": y_pred,
        "Predicted Probability": np.round(y_proba, 4),
    })
    df_preds.to_csv(paths["pred_output"], index=False, encoding="utf-8")
    print(f"[OK] Saved Predictions to: {paths['pred_output'].name}")

    # 7. 5-Fold Stratified Cross-Validation on Training Data ONLY (Task 7)
    print("\n[3] Running 5-Fold Stratified Cross-Validation strictly on X_train (N=2,400)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC-AUC": []}

    for fold_idx, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr_f, X_val_f = X_train.iloc[t_idx], X_train.iloc[v_idx]
        y_tr_f, y_val_f = y_train.iloc[t_idx], y_train.iloc[v_idx]

        fold_scale_w = float((y_tr_f == 0).sum() / (y_tr_f == 1).sum())
        fold_xgb = XGBClassifier(
            objective=OBJECTIVE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            scale_pos_weight=fold_scale_w,
            random_state=RANDOM_STATE,
            eval_metric=EVAL_METRIC,
            n_jobs=N_JOBS,
        )
        fold_xgb.fit(X_tr_f, y_tr_f)
        p_fold = fold_xgb.predict(X_val_f)
        prob_fold = fold_xgb.predict_proba(X_val_f)[:, 1]

        cv_metrics["Accuracy"].append(accuracy_score(y_val_f, p_fold))
        cv_metrics["Precision"].append(precision_score(y_val_f, p_fold, zero_division=0))
        cv_metrics["Recall"].append(recall_score(y_val_f, p_fold, zero_division=0))
        cv_metrics["F1-Score"].append(f1_score(y_val_f, p_fold, zero_division=0))
        cv_metrics["ROC-AUC"].append(roc_auc_score(y_val_f, prob_fold))

    cv_summary_records = []
    for metric_name in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        mean_val = float(np.mean(cv_metrics[metric_name]))
        std_val = float(np.std(cv_metrics[metric_name]))
        cv_summary_records.append({
            "Metric": metric_name,
            "CV_Mean": round(mean_val, 4),
            "CV_Std": round(std_val, 4),
            "Fold_1": round(cv_metrics[metric_name][0], 4),
            "Fold_2": round(cv_metrics[metric_name][1], 4),
            "Fold_3": round(cv_metrics[metric_name][2], 4),
            "Fold_4": round(cv_metrics[metric_name][3], 4),
            "Fold_5": round(cv_metrics[metric_name][4], 4),
        })

    df_cv = pd.DataFrame(cv_summary_records)
    df_cv.to_csv(paths["cv_output"], index=False, encoding="utf-8")
    print(f"[OK] Saved Cross-Validation results to: {paths['cv_output'].name}")
    print(df_cv.to_string(index=False))

    # 8. Native Feature Importance (Task 10)
    print("\n[4] Extracting native Gain feature importances...")
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Gain_Importance": xgb_model.feature_importances_,
        "Normalized_Importance_%": (xgb_model.feature_importances_ / xgb_model.feature_importances_.sum()) * 100,
    }).sort_values(by="Gain_Importance", ascending=False)

    df_imp.to_csv(paths["imp_output"], index=False, encoding="utf-8")
    print(f"[OK] Saved Feature Importance to: {paths['imp_output'].name}")
    print(df_imp.head(10).to_string(index=False))

    # 9. ROC Curve (Task 8)
    print("\n[5] Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color="#ff7f0e", lw=2, label=f"Final XGBoost Proposed (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve\nFinal XGBoost Proposed Model (Dataset v2, N=600)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(paths["roc_output"])
    plt.close()
    print(f"[OK] Saved ROC Curve to: {paths['roc_output'].name}")

    # 10. Precision-Recall Curve (Task 9)
    print("[6] Plotting Precision-Recall Curve...")
    prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_proba)
    prevalence = float((y_test == 1).sum() / len(y_test))
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(rec_pts, prec_pts, color="#ff7f0e", lw=2, label=f"Final XGBoost Proposed (AP = {avg_prec:.4f})")
    plt.axhline(y=prevalence, color="grey", linestyle="--", lw=1.5, label=f"No-Skill Baseline (Prevalence = {prevalence:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve\nFinal XGBoost Proposed Model (Dataset v2, N=600)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(paths["pr_output"])
    plt.close()
    print(f"[OK] Saved Precision-Recall Curve to: {paths['pr_output'].name}")

    # 11. Results Summary JSON (Task 3)
    results_summary = {
        "dataset_name": DATASET_NAME,
        "dataset_path": "data/synthetic/loan_evaluation_dataset_3000_v2.csv",
        "dataset_size": 3000,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(feature_names),
        "scale_pos_weight": float(scale_pos_weight),
        "target_distribution": {
            "Approved": int((y_train == 1).sum() + (y_test == 1).sum()),
            "Rejected": int((y_train == 0).sum() + (y_test == 0).sum()),
        },
        "model_configuration": {
            "model_type": "XGBClassifier",
            "objective": OBJECTIVE,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "scale_pos_weight": float(scale_pos_weight),
            "random_state": RANDOM_STATE,
            "eval_metric": EVAL_METRIC,
            "n_jobs": N_JOBS,
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision": round(avg_prec, 4),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "cross_validation_5_fold": {
            "accuracy_mean": round(float(np.mean(cv_metrics["Accuracy"])), 4),
            "accuracy_std": round(float(np.std(cv_metrics["Accuracy"])), 4),
            "precision_mean": round(float(np.mean(cv_metrics["Precision"])), 4),
            "precision_std": round(float(np.std(cv_metrics["Precision"])), 4),
            "recall_mean": round(float(np.mean(cv_metrics["Recall"])), 4),
            "recall_std": round(float(np.std(cv_metrics["Recall"])), 4),
            "f1_mean": round(float(np.mean(cv_metrics["F1-Score"])), 4),
            "f1_std": round(float(np.std(cv_metrics["F1-Score"])), 4),
            "roc_auc_mean": round(float(np.mean(cv_metrics["ROC-AUC"])), 4),
            "roc_auc_std": round(float(np.std(cv_metrics["ROC-AUC"])), 4),
        },
        "top_10_features_gain": df_imp.head(10)[["Feature", "Gain_Importance"]].to_dict(orient="records"),
    }

    with open(paths["results_json"], "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[OK] Saved comprehensive Results JSON to: {paths['results_json'].name}")

    # 12. Final Model Comparison (Task 11)
    print("\n[7] Generating Final Model Comparison (Random Forest vs. XGBoost)...")
    if paths["rf_results"].exists():
        with open(paths["rf_results"], "r", encoding="utf-8") as f:
            rf_res = json.load(f)

        rf_tm = rf_res["test_metrics"]
        xgb_tm = results_summary["test_metrics"]

        comp_records = [
            {
                "Model": "Random Forest (Final Baseline)",
                "Accuracy": rf_tm["accuracy"],
                "Precision": rf_tm["precision"],
                "Recall": rf_tm["recall"],
                "F1-Score": rf_tm["f1_score"],
                "ROC-AUC": rf_tm["roc_auc"],
                "Average_Precision": rf_tm["average_precision"],
            },
            {
                "Model": "XGBoost (Final Proposed)",
                "Accuracy": xgb_tm["accuracy"],
                "Precision": xgb_tm["precision"],
                "Recall": xgb_tm["recall"],
                "F1-Score": xgb_tm["f1_score"],
                "ROC-AUC": xgb_tm["roc_auc"],
                "Average_Precision": xgb_tm["average_precision"],
            },
            {
                "Model": "Absolute Difference (XGB - RF)",
                "Accuracy": round(xgb_tm["accuracy"] - rf_tm["accuracy"], 4),
                "Precision": round(xgb_tm["precision"] - rf_tm["precision"], 4),
                "Recall": round(xgb_tm["recall"] - rf_tm["recall"], 4),
                "F1-Score": round(xgb_tm["f1_score"] - rf_tm["f1_score"], 4),
                "ROC-AUC": round(xgb_tm["roc_auc"] - rf_tm["roc_auc"], 4),
                "Average_Precision": round(xgb_tm["average_precision"] - rf_tm["average_precision"], 4),
            },
        ]
        df_comp = pd.DataFrame(comp_records)
        df_comp.to_csv(paths["comparison_csv"], index=False, encoding="utf-8")
        print(f"[OK] Saved Final Comparison to: {paths['comparison_csv'].name}")
        print("\n--- FINAL MODEL COMPARISON SUMMARY ---")
        print(df_comp.to_string(index=False))

    print("\n" + "=" * 85)
    print("[SUCCESS] REVISED STEP 10 FINAL XGBOOST EXECUTION COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
