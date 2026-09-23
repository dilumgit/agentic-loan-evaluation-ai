"""
Validation and Model-Readiness Test for Dataset v2 (3,000 Records, Gaussian Copula).

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Task 10: Compare Original 500 vs. Old CTGAN 3,000 vs. New Gaussian Copula 3,000 v2.
- Task 11: Lightweight Model-Readiness Cross-Validation Test (RF and XGBoost).
"""

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_paths() -> Dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent if script_dir.name == "data" else script_dir

    return {
        "seed_500": workspace_root / "data" / "raw" / "google_form_responses_500.csv",
        "old_ctgan_3000": workspace_root / "data" / "synthetic" / "loan_evaluation_dataset_3000.csv",
        "new_v2_3000": workspace_root / "data" / "synthetic" / "loan_evaluation_dataset_3000_v2.csv",
        "eval_dir": workspace_root / "evaluation",
    }


def compute_associations(df: pd.DataFrame, target: str = "Loan Application Result") -> Dict[str, float]:
    features = [c for c in df.columns if c != target]
    res = {}
    for f in features:
        ct = pd.crosstab(df[f], df[target])
        chi2, p, dof, ex = stats.chi2_contingency(ct)
        n = len(df)
        k = min(ct.shape)
        v = float(np.sqrt(chi2 / (n * (k - 1))))
        res[f] = round(v, 4)
    return res


def run_validation():
    print("=" * 85)
    print("STEP 7 — TASK 10 & 11: DATASET v2 VALIDATION & MODEL-READINESS DIAGNOSTIC")
    print("=" * 85)

    paths = resolve_paths()

    df_seed = pd.read_csv(paths["seed_500"])
    df_old = pd.read_csv(paths["old_ctgan_3000"])
    df_new = pd.read_csv(paths["new_v2_3000"])

    target = "Loan Application Result"
    features = [c for c in df_seed.columns if c != target]

    print("\n--- 1. DATASET PROPERTIES & INTEGRITY CHECKS ---")
    print(f"Original Seed 500  : Shape = {df_seed.shape} | Nulls = {df_seed.isnull().sum().sum()} | Duplicates = {df_seed.duplicated().sum()}")
    print(f"Old CTGAN 3000     : Shape = {df_old.shape} | Nulls = {df_old.isnull().sum().sum()} | Duplicates = {df_old.duplicated().sum()}")
    print(f"New Copula 3000 v2 : Shape = {df_new.shape} | Nulls = {df_new.isnull().sum().sum()} | Duplicates = {df_new.duplicated().sum()}")

    print("\n--- 2. TARGET DISTRIBUTION COMPARISON ---")
    def target_summary(df, name):
        app = (df[target] == "Approved").sum()
        rej = (df[target] == "Rejected").sum()
        total = len(df)
        return {
            "Dataset": name,
            "Total_Rows": total,
            "Approved_Count": app,
            "Approved_%": round(app / total * 100, 2),
            "Rejected_Count": rej,
            "Rejected_%": round(rej / total * 100, 2),
        }

    df_target_comp = pd.DataFrame([
        target_summary(df_seed, "Original Seed (500)"),
        target_summary(df_old, "Old CTGAN Combined (3,000)"),
        target_summary(df_new, "New GaussianCopula v2 (3,000)"),
    ])
    print(df_target_comp.to_string(index=False))

    print("\n--- 3. FEATURE-TARGET ASSOCIATION (CRAMER'S V) COMPARISON ---")
    v_seed = compute_associations(df_seed, target)
    v_old = compute_associations(df_old, target)
    v_new = compute_associations(df_new, target)

    v_rows = []
    for f in features:
        v_rows.append({
            "Feature": f,
            "Seed_V (500)": v_seed[f],
            "Old_CTGAN_V (3000)": v_old[f],
            "New_Copula_V2 (3000)": v_new[f],
            "Delta_v2_vs_Old": round(v_new[f] - v_old[f], 4),
        })
    df_v_summary = pd.DataFrame(v_rows).sort_values(by="Seed_V (500)", ascending=False)
    print(df_v_summary.to_string(index=False))

    print(f"\nMean Cramér's V Across All 16 Features:")
    print(f"  Seed (500)          : {np.mean(list(v_seed.values())):.4f}")
    print(f"  Old CTGAN (3,000)   : {np.mean(list(v_old.values())):.4f}")
    print(f"  New Copula v2 (3000): {np.mean(list(v_new.values())):.4f} (+{round((np.mean(list(v_new.values())) - np.mean(list(v_old.values()))) * 100, 2)}% higher association)")

    # 4. Model-Readiness Test
    print("\n--- 4. LIGHTWEIGHT MODEL-READINESS DIAGNOSTIC (5-FOLD CV) ---")
    ORDINAL_FEATURES = [
        "Age Group",
        "Number of Dependents",
        "Employment Duration",
        "Monthly Income",
        "Monthly Loan Repayment",
        "Requested Loan Amount",
    ]
    ORDINAL_CATEGORIES = [
        ["Below 25", "25–34", "35–44", "45–54", "55–64", "65 or above"],
        ["0", "1", "2", "3", "4", "5 or more"],
        ["Not employed/business", "Less than 1 year", "1–2 years", "3–5 years", "6–10 years", "More than 10 years"],
        [
            "Below Rs.50,000",
            "Rs.50,000–99,999",
            "Rs.100,000–149,999",
            "Rs.150,000–249,999",
            "Rs.250,000–399,999",
            "Rs.400,000–599,999",
            "Rs.600,000–999,999",
            "Rs.1,000,000 or above",
        ],
        [
            "No existing loan",
            "Below Rs.10,000",
            "Rs.10,000–24,999",
            "Rs.25,000–49,999",
            "Rs.50,000–74,999",
            "Rs.75,000–99,999",
            "Rs.100,000+",
        ],
        [
            "Below Rs.500,000",
            "Rs.500,000–999,999",
            "Rs.1,000,000–4,999,999",
            "Rs.5,000,000–9,999,999",
            "Rs.10,000,000–24,999,999",
            "Rs.25,000,000–49,999,999",
            "Rs.50,000,000–74,999,999",
            "Rs.75,000,000–99,999,999",
            "Rs.100,000,000 or above",
        ],
    ]
    NOMINAL_FEATURES = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Employment Type",
        "Existing Loan Status",
        "Loan Type",
        "Loan Purpose",
        "Collateral Availability",
        "Guarantor Availability",
        "Required Documents Submitted",
    ]

    for d_name, d_df in [("Old CTGAN 3,000 Dataset", df_old), ("New GaussianCopula 3,000 v2 Dataset", df_new)]:
        X = d_df.drop(columns=[target])
        y = d_df[target].map({"Approved": 1, "Rejected": 0})

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    ORDINAL_FEATURES,
                ),
                ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        X_enc = preprocessor.fit_transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rf_aucs, rf_accs, rf_f1s = [], [], []
        xgb_aucs, xgb_accs, xgb_f1s = [], [], []

        for t_idx, v_idx in skf.split(X_enc, y):
            X_tr, X_val = X_enc[t_idx], X_enc[v_idx]
            y_tr, y_val = y.iloc[t_idx], y.iloc[v_idx]

            # RF
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf = rf.predict(X_val)
            prob_rf = rf.predict_proba(X_val)[:, 1]
            rf_aucs.append(roc_auc_score(y_val, prob_rf))
            rf_accs.append(accuracy_score(y_val, p_rf))
            rf_f1s.append(f1_score(y_val, p_rf, zero_division=0))

            # XGB
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
            xgb_aucs.append(roc_auc_score(y_val, prob_xgb))
            xgb_accs.append(accuracy_score(y_val, p_xgb))
            xgb_f1s.append(f1_score(y_val, p_xgb, zero_division=0))

        print(f"\n[{d_name}] 5-Fold Stratified Cross-Validation Results:")
        print(f"  Random Forest : Accuracy = {np.mean(rf_accs):.4f} +/- {np.std(rf_accs):.4f} | F1 = {np.mean(rf_f1s):.4f} | ROC-AUC = {np.mean(rf_aucs):.4f} +/- {np.std(rf_aucs):.4f}")
        print(f"  XGBoost       : Accuracy = {np.mean(xgb_accs):.4f} +/- {np.std(xgb_accs):.4f} | F1 = {np.mean(xgb_f1s):.4f} | ROC-AUC = {np.mean(xgb_aucs):.4f} +/- {np.std(xgb_aucs):.4f}")

    print("\n" + "=" * 85)
    print("[SUCCESS] VALIDATION & MODEL-READINESS DIAGNOSTIC COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_validation()
