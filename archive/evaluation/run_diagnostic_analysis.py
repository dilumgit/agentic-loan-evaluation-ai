"""
Model Diagnostic Analysis and Statistical Validation Pipeline.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Task 1: Compare categorical distributions (500 Seed vs. 2,500 SDV).
- Task 2: Analyze Target distribution shift and mode concentration.
- Task 3: Compute Contingency tables, Chi-square statistics, and Cramér's V associations.
- Task 4: Execute 5-Fold Stratified Cross-Validation on X_train (2,400 samples) for RF and XGBoost.
- Task 5: Plot and save ROC Curves on the holdout test set (600 samples).
- Task 6: Plot and save Precision-Recall Curves on the holdout test set.
- Task 7: Generate statistical analysis of predicted probability distributions.
- Task 8: Extract and save native feature importances from RF and XGBoost models.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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


def resolve_paths() -> Dict[str, Path]:
    """Resolve paths to all data, model, and evaluation resources."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    return {
        "seed_csv": workspace_root / "data" / "raw" / "google_form_responses_500.csv",
        "synth_csv": workspace_root / "data" / "synthetic" / "sdv_generated_2500.csv",
        "comb_csv": workspace_root / "data" / "synthetic" / "loan_evaluation_dataset_3000.csv",
        "X_train": workspace_root / "data" / "processed" / "X_train.csv",
        "X_test": workspace_root / "data" / "processed" / "X_test.csv",
        "y_train": workspace_root / "data" / "processed" / "y_train.csv",
        "y_test": workspace_root / "data" / "processed" / "y_test.csv",
        "feature_names": workspace_root / "data" / "processed" / "feature_names.json",
        "rf_model": workspace_root / "models" / "baseline" / "random_forest.joblib",
        "xgb_model": workspace_root / "models" / "xgboost" / "xgboost_model.joblib",
        "eval_dir": workspace_root / "evaluation",
    }


def task1_original_vs_sdv_distribution(paths: Dict[str, Path]) -> pd.DataFrame:
    """Compare categorical distributions between 500 Seed and 2,500 SDV records."""
    print("\n" + "=" * 80)
    print("TASK 1 & 2: ORIGINAL VS SDV DISTRIBUTION & TARGET SHIFT ANALYSIS")
    print("=" * 80)

    try:
        df_seed = pd.read_csv(paths["seed_csv"], encoding="utf-8")
    except UnicodeDecodeError:
        df_seed = pd.read_csv(paths["seed_csv"], encoding="cp1252")

    try:
        df_synth = pd.read_csv(paths["synth_csv"], encoding="utf-8")
    except UnicodeDecodeError:
        df_synth = pd.read_csv(paths["synth_csv"], encoding="cp1252")

    records = []
    for col in df_seed.columns:
        s_vc = df_seed[col].value_counts()
        g_vc = df_synth[col].value_counts()
        all_cats = sorted(list(set(s_vc.index).union(set(g_vc.index))))
        for c in all_cats:
            s_cnt = int(s_vc.get(c, 0))
            s_pct = round(s_cnt / len(df_seed) * 100, 2)
            g_cnt = int(g_vc.get(c, 0))
            g_pct = round(g_cnt / len(df_synth) * 100, 2)
            pp_diff = round(g_pct - s_pct, 2)
            abs_pp_diff = round(abs(g_pct - s_pct), 2)
            records.append({
                "Feature": col,
                "Category": c,
                "Seed_Count_500": s_cnt,
                "Seed_Pct": s_pct,
                "SDV_Count_2500": g_cnt,
                "SDV_Pct": g_pct,
                "Percentage_Point_Diff": pp_diff,
                "Abs_Percentage_Point_Diff": abs_pp_diff,
            })

    df_dist = pd.DataFrame(records)
    out_path = paths["eval_dir"] / "original_vs_sdv_distribution.csv"
    df_dist.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved original vs SDV distribution to: {out_path.name}")
    return df_dist


def task3_feature_target_association(paths: Dict[str, Path]) -> pd.DataFrame:
    """Calculate Chi-square test of independence and Cramér's V association."""
    print("\n" + "=" * 80)
    print("TASK 3: FEATURE-TARGET STATISTICAL ASSOCIATION (CHI-SQUARE & CRAMER'S V)")
    print("=" * 80)

    try:
        df = pd.read_csv(paths["comb_csv"], encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(paths["comb_csv"], encoding="cp1252")

    target = "Loan Application Result"
    features = [c for c in df.columns if c != target]

    assoc_records = []
    for feat in features:
        ct = pd.crosstab(df[feat], df[target])
        chi2, p_val, dof, ex = stats.chi2_contingency(ct)
        n = len(df)
        k = min(ct.shape)
        cramers_v = float(np.sqrt(chi2 / (n * (k - 1))))
        assoc_records.append({
            "Feature": feat,
            "Categories": int(ct.shape[0]),
            "Chi2_Statistic": round(float(chi2), 4),
            "p_value": round(float(p_val), 6),
            "Degrees_of_Freedom": int(dof),
            "Cramers_V": round(cramers_v, 4),
            "Statistically_Significant (p<0.05)": bool(p_val < 0.05),
        })

    df_assoc = pd.DataFrame(assoc_records).sort_values(by="Cramers_V", ascending=False)
    out_path = paths["eval_dir"] / "feature_target_association.csv"
    df_assoc.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved feature-target associations to: {out_path.name}")
    print(df_assoc.to_string(index=False))
    return df_assoc


def task4_model_cross_validation(paths: Dict[str, Path]) -> pd.DataFrame:
    """Run 5-Fold Stratified Cross-Validation on X_train (2,400 samples)."""
    print("\n" + "=" * 80)
    print("TASK 4: 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING DATA ONLY (N=2,400)")
    print("=" * 80)

    X_train = pd.read_csv(paths["X_train"])
    y_train = pd.read_csv(paths["y_train"]).iloc[:, 0]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_folds = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC-AUC": []}
    xgb_folds = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC-AUC": []}

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[t_idx], X_train.iloc[v_idx]
        y_tr, y_val = y_train.iloc[t_idx], y_train.iloc[v_idx]

        # 1. Random Forest Fold
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict(X_val)
        prob_rf = rf.predict_proba(X_val)[:, 1]

        rf_folds["Accuracy"].append(accuracy_score(y_val, p_rf))
        rf_folds["Precision"].append(precision_score(y_val, p_rf, zero_division=0))
        rf_folds["Recall"].append(recall_score(y_val, p_rf, zero_division=0))
        rf_folds["F1-Score"].append(f1_score(y_val, p_rf, zero_division=0))
        rf_folds["ROC-AUC"].append(roc_auc_score(y_val, prob_rf))

        # 2. XGBoost Fold
        scale_w = float((y_tr == 0).sum() / (y_tr == 1).sum())
        xgb = XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_w,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb.fit(X_tr, y_tr)
        p_xgb = xgb.predict(X_val)
        prob_xgb = xgb.predict_proba(X_val)[:, 1]

        xgb_folds["Accuracy"].append(accuracy_score(y_val, p_xgb))
        xgb_folds["Precision"].append(precision_score(y_val, p_xgb, zero_division=0))
        xgb_folds["Recall"].append(recall_score(y_val, p_xgb, zero_division=0))
        xgb_folds["F1-Score"].append(f1_score(y_val, p_xgb, zero_division=0))
        xgb_folds["ROC-AUC"].append(roc_auc_score(y_val, prob_xgb))

    cv_records = []
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        rf_mean = float(np.mean(rf_folds[metric]))
        rf_std = float(np.std(rf_folds[metric]))
        xgb_mean = float(np.mean(xgb_folds[metric]))
        xgb_std = float(np.std(xgb_folds[metric]))
        cv_records.append({
            "Metric": metric,
            "RF_CV_Mean": round(rf_mean, 4),
            "RF_CV_Std": round(rf_std, 4),
            "XGB_CV_Mean": round(xgb_mean, 4),
            "XGB_CV_Std": round(xgb_std, 4),
            "Delta_Mean (XGB - RF)": round(xgb_mean - rf_mean, 4),
        })

    df_cv = pd.DataFrame(cv_records)
    out_path = paths["eval_dir"] / "cross_validation_results.csv"
    df_cv.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved 5-fold cross-validation results to: {out_path.name}")
    print(df_cv.to_string(index=False))
    return df_cv


def task5_6_roc_pr_curves(paths: Dict[str, Path]) -> None:
    """Generate and persist ROC and Precision-Recall Curves."""
    print("\n" + "=" * 80)
    print("TASK 5 & 6: GENERATING ROC AND PRECISION-RECALL CURVES (HOLD-OUT N=600)")
    print("=" * 80)

    rf_model = joblib.load(paths["rf_model"])
    xgb_model = joblib.load(paths["xgb_model"])

    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]

    p_rf = rf_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # 1. ROC Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, p_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, p_xgb)
    auc_rf = roc_auc_score(y_test, p_rf)
    auc_xgb = roc_auc_score(y_test, p_xgb)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_rf, tpr_rf, color="#1f77b4", lw=2, label=f"Random Forest Baseline (AUC = {auc_rf:.4f})")
    plt.plot(fpr_xgb, tpr_xgb, color="#ff7f0e", lw=2, label=f"XGBoost Proposed (AUC = {auc_xgb:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve Comparison\n(3,000-Record Dataset Holdout Test Set, N=600)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_path = paths["eval_dir"] / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"[OK] Saved ROC Curve to: {roc_path.name}")

    # 2. Precision-Recall Curves
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, p_rf)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, p_xgb)
    ap_rf = average_precision_score(y_test, p_rf)
    ap_xgb = average_precision_score(y_test, p_xgb)
    prevalence = float((y_test == 1).sum() / len(y_test))

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(rec_rf, prec_rf, color="#1f77b4", lw=2, label=f"Random Forest Baseline (AP = {ap_rf:.4f})")
    plt.plot(rec_xgb, prec_xgb, color="#ff7f0e", lw=2, label=f"XGBoost Proposed (AP = {ap_xgb:.4f})")
    plt.axhline(y=prevalence, color="grey", linestyle="--", lw=1.5, label=f"No-Skill Prevalence Line (P = {prevalence:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (True Positive Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve Comparison\n(3,000-Record Dataset Holdout Test Set, N=600)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    pr_path = paths["eval_dir"] / "precision_recall_curve.png"
    plt.savefig(pr_path)
    plt.close()
    print(f"[OK] Saved Precision-Recall Curve to: {pr_path.name}")


def task7_prediction_probability_analysis(paths: Dict[str, Path]) -> pd.DataFrame:
    """Analyze distribution of predicted probabilities across both models."""
    print("\n" + "=" * 80)
    print("TASK 7: MODEL PREDICTED PROBABILITY DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)

    rf_model = joblib.load(paths["rf_model"])
    xgb_model = joblib.load(paths["xgb_model"])

    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]

    p_rf = rf_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    def stats_dict(arr, partition_name, model_name):
        return {
            "Model": model_name,
            "Cohort": partition_name,
            "Sample_Count": len(arr),
            "Mean_Probability": round(float(np.mean(arr)), 4),
            "Std_Dev": round(float(np.std(arr)), 4),
            "Min": round(float(np.min(arr)), 4),
            "Q25": round(float(np.percentile(arr, 25)), 4),
            "Median": round(float(np.median(arr)), 4),
            "Q75": round(float(np.percentile(arr, 75)), 4),
            "Max": round(float(np.max(arr)), 4),
        }

    recs = [
        stats_dict(p_rf, "All Test Samples (N=600)", "Random Forest"),
        stats_dict(p_rf[y_test == 1], "Actual Approved (N=209)", "Random Forest"),
        stats_dict(p_rf[y_test == 0], "Actual Rejected (N=391)", "Random Forest"),
        stats_dict(p_xgb, "All Test Samples (N=600)", "XGBoost"),
        stats_dict(p_xgb[y_test == 1], "Actual Approved (N=209)", "XGBoost"),
        stats_dict(p_xgb[y_test == 0], "Actual Rejected (N=391)", "XGBoost"),
    ]

    df_prob = pd.DataFrame(recs)
    out_path = paths["eval_dir"] / "prediction_probability_analysis.csv"
    df_prob.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved prediction probability analysis to: {out_path.name}")
    print(df_prob.to_string(index=False))
    return df_prob


def task8_feature_importance(paths: Dict[str, Path]) -> None:
    """Extract native feature importances from RF and XGBoost."""
    print("\n" + "=" * 80)
    print("TASK 8: NATIVE FEATURE IMPORTANCE EXTRACTION (RF & XGBOOST)")
    print("=" * 80)

    with open(paths["feature_names"], "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    rf_model = joblib.load(paths["rf_model"])
    xgb_model = joblib.load(paths["xgb_model"])

    df_rf_imp = pd.DataFrame({
        "Feature": feature_names,
        "Gini_Importance": rf_model.feature_importances_,
        "Normalized_Importance_%": (rf_model.feature_importances_ / rf_model.feature_importances_.sum()) * 100,
    }).sort_values(by="Gini_Importance", ascending=False)

    df_xgb_imp = pd.DataFrame({
        "Feature": feature_names,
        "Gain_Importance": xgb_model.feature_importances_,
        "Normalized_Importance_%": (xgb_model.feature_importances_ / xgb_model.feature_importances_.sum()) * 100,
    }).sort_values(by="Gain_Importance", ascending=False)

    rf_out = paths["eval_dir"] / "random_forest_feature_importance.csv"
    xgb_out = paths["eval_dir"] / "xgboost_feature_importance.csv"

    df_rf_imp.to_csv(rf_out, index=False, encoding="utf-8")
    df_xgb_imp.to_csv(xgb_out, index=False, encoding="utf-8")

    print(f"[OK] Saved Random Forest feature importances to: {rf_out.name}")
    print(f"[OK] Saved XGBoost feature importances to: {xgb_out.name}")


def main():
    paths = resolve_paths()
    task1_original_vs_sdv_distribution(paths)
    task3_feature_target_association(paths)
    task4_model_cross_validation(paths)
    task5_6_roc_pr_curves(paths)
    task7_prediction_probability_analysis(paths)
    task8_feature_importance(paths)
    print("\n" + "=" * 80)
    print("[SUCCESS] ALL DIAGNOSTIC CALCULATIONS EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
