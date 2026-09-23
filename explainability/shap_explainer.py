"""
SHAP (TreeSHAP) Explainability Pipeline for XGBoost Loan Decision Support System.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Dataset:
- Dataset v2: loan_evaluation_dataset_3000_v2.csv
- Preprocessed: data/processed/ (X_test: 600 x 55)
- Model: models/xgboost/final_xgboost.joblib (Frozen)

Module Purpose:
- Implements TreeSHAP algorithm for global and local interpretability.
- Computes mean absolute Shapley values across 600 holdout instances.
- Generates beeswarm summary plot and global bar plot.
- Generates local feature attribution for 4 representative cases (TP, TN, FP, FN).
- Saves all artifacts to evaluation/.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


REPRESENTATIVE_CASES = [
    {"case_type": "True Positive (Approved -> Approved)", "index": 39, "actual_label": 1},
    {"case_type": "True Negative (Rejected -> Rejected)", "index": 460, "actual_label": 0},
    {"case_type": "False Positive (Rejected -> Approved)", "index": 503, "actual_label": 0},
    {"case_type": "False Negative (Approved -> Rejected)", "index": 461, "actual_label": 1},
]


def resolve_paths() -> Dict[str, Path]:
    """Resolve directory and file paths."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    data_dir = workspace_root / "data" / "processed"
    models_dir = workspace_root / "models" / "xgboost"
    eval_dir = workspace_root / "evaluation"

    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model": models_dir / "final_xgboost.joblib",
        "X_test": data_dir / "X_test.csv",
        "y_test": data_dir / "y_test.csv",
        "feature_names": data_dir / "feature_names.json",
        "global_imp_csv": eval_dir / "shap_global_importance.csv",
        "summary_png": eval_dir / "shap_summary.png",
        "bar_png": eval_dir / "shap_feature_importance.png",
        "local_csv": eval_dir / "shap_local_examples.csv",
        "local_png": eval_dir / "shap_local_example.png",
    }


def run_shap_analysis() -> Dict[str, Any]:
    print("=" * 85)
    print("STEP 12: PART A — SHAP EXPLAINABILITY PIPELINE (TREE-SHAP)")
    print("=" * 85)

    paths = resolve_paths()

    # 1. Load Data and Frozen Model
    model = joblib.load(paths["model"])
    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]

    with open(paths["feature_names"], "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"Loaded Frozen XGBoost Model : {paths['model'].name}")
    print(f"Loaded Test Feature Matrix  : {X_test.shape}")
    print(f"Feature Names Count         : {len(feature_names)}")

    assert X_test.shape == (600, 55), f"Unexpected X_test shape: {X_test.shape}"
    assert len(feature_names) == 55, f"Unexpected feature names count: {len(feature_names)}"

    # 2. Compute TreeSHAP Values
    print("\n[1] Initializing TreeExplainer and calculating Shapley values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = float(explainer.expected_value)

    print(f"[OK] Computed SHAP values matrix of shape: {shap_values.shape}")
    print(f"[OK] Base Value (Expected Margin): {expected_value:.4f}")

    assert not np.isnan(shap_values).any(), "Found NaN in SHAP values!"
    assert not np.isinf(shap_values).any(), "Found Inf in SHAP values!"

    # 3. Global SHAP Feature Importance
    print("\n[2] Computing Global Mean Absolute SHAP Importances...")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = float(np.sum(mean_abs_shap))

    df_global = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Normalized_Importance_%": (mean_abs_shap / total_shap) * 100,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)

    df_global.to_csv(paths["global_imp_csv"], index=False, encoding="utf-8")
    print(f"[OK] Saved Global SHAP Importance to: {paths['global_imp_csv'].name}")
    print("\n--- TOP 10 GLOBAL SHAP FEATURES ---")
    print(df_global.head(10).to_string(index=False))

    # 4. Global Visualizations
    print("\n[3] Generating SHAP Summary Beeswarm Plot...")
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("SHAP Beeswarm Summary Plot\nGlobal Feature Attributions (Dataset v2, N=600)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(paths["summary_png"], bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP Beeswarm Plot to: {paths['summary_png'].name}")

    print("[4] Generating SHAP Global Feature Importance Bar Plot...")
    top_15_df = df_global.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7), dpi=300)
    plt.barh(top_15_df["Feature"], top_15_df["Mean_Abs_SHAP"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=11, fontweight="bold")
    plt.title("Global Feature Importance (TreeSHAP)\nTop 15 Predictor Attributes", fontsize=12, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.tight_layout()
    plt.savefig(paths["bar_png"], bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP Bar Plot to: {paths['bar_png'].name}")

    # 5. Local Explanations for Representative Cases
    print("\n[5] Generating Local SHAP Explanations for 4 Representative Test Cases...")
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    local_records = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    for idx, case_info in enumerate(REPRESENTATIVE_CASES):
        t_idx = case_info["index"]
        c_type = case_info["case_type"]
        act_lbl = int(y_test.iloc[t_idx])
        pred_lbl = int(preds[t_idx])
        prob_val = float(probas[t_idx])

        row_shap = shap_values[t_idx]
        df_row = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": row_shap,
            "Feature_Value": X_test.iloc[t_idx].values,
        })

        pos_factors = df_row.sort_values(by="SHAP_Value", ascending=False).head(3)
        neg_factors = df_row.sort_values(by="SHAP_Value", ascending=True).head(3)

        pos_str = "; ".join([f"{r['Feature']} (+{r['SHAP_Value']:.3f})" for _, r in pos_factors.iterrows()])
        neg_str = "; ".join([f"{r['Feature']} ({r['SHAP_Value']:.3f})" for _, r in neg_factors.iterrows()])

        local_records.append({
            "Case_Type": c_type,
            "Test_Index": t_idx,
            "Actual_Class": "Approved (1)" if act_lbl == 1 else "Rejected (0)",
            "Predicted_Class": "Approved (1)" if pred_lbl == 1 else "Rejected (0)",
            "Predicted_Probability": round(prob_val, 4),
            "Top_Positive_Contributing_Features": pos_str,
            "Top_Negative_Contributing_Features": neg_str,
        })

        # Plot local horizontal bar for this case
        top_local = pd.concat([pos_factors, neg_factors]).drop_duplicates().sort_values(by="SHAP_Value", ascending=True)
        colors = ["#2ca02c" if val > 0 else "#d62728" for val in top_local["SHAP_Value"]]

        ax = axes[idx]
        ax.barh(top_local["Feature"], top_local["SHAP_Value"], color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.set_title(f"{c_type}\nTest Index: {t_idx} | P(Approved)={prob_val:.4f} | Actual: {act_lbl}", fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Value (Contribution to Log-Odds)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    plt.savefig(paths["local_png"], bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Local SHAP Multi-panel Plot to: {paths['local_png'].name}")

    df_local = pd.DataFrame(local_records)
    df_local.to_csv(paths["local_csv"], index=False, encoding="utf-8")
    print(f"[OK] Saved Local SHAP Summary Table to: {paths['local_csv'].name}")
    print("\n--- REPRESENTATIVE LOCAL SHAP EXPLANATIONS ---")
    print(df_local.to_string(index=False))

    print("\n" + "=" * 85)
    print("[SUCCESS] SHAP EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 85)

    return {
        "shap_values": shap_values,
        "global_importance": df_global,
        "local_examples": df_local,
    }


if __name__ == "__main__":
    run_shap_analysis()
