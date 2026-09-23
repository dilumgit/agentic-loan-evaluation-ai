"""
Dataset Preprocessing Pipeline for Bank Loan Evaluation Academic Research (Dataset v2).

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Loads the primary 3,000-record research dataset v2 (500 Google Form seed + 2,500 Gaussian Copula synthetic records).
- Validates data integrity (3,000 rows, 17 columns, 0 missing values, 0 duplicates).
- Encodes target variable: Approved = 1, Rejected = 0.
- Performs stratified train/test split (80/20, random_state=42, stratify=y).
- Builds and fits a scikit-learn ColumnTransformer on the training data ONLY:
  * OrdinalEncoder for 6 features with natural hierarchical orders.
  * OneHotEncoder for 10 nominal categorical features.
- Transforms training (2,400 samples) and test (600 samples) data without data leakage.
- Persists processed datasets, feature names, preprocessor artifact, and evaluation/final_dataset_v2_summary.json.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# Configuration & Schema Definitions
# -----------------------------------------------------------------------------

DATASET_NAME: str = "loan_evaluation_dataset_3000_v2.csv"
SELECTED_SYNTHESIZER: str = "GaussianCopulaSynthesizer"

TARGET_COLUMN: str = "Loan Application Result"
TARGET_MAPPING: Dict[str, int] = {
    "Approved": 1,
    "Rejected": 0,
}

EXPECTED_PREDICTORS: List[str] = [
    "Age Group",
    "Gender",
    "Marital Status",
    "Number of Dependents",
    "Education Level",
    "Employment Type",
    "Employment Duration",
    "Monthly Income",
    "Existing Loan Status",
    "Monthly Loan Repayment",
    "Loan Type",
    "Requested Loan Amount",
    "Loan Purpose",
    "Collateral Availability",
    "Guarantor Availability",
    "Required Documents Submitted",
]

# Explicit hierarchy for Ordinal Features
ORDINAL_FEATURES: List[str] = [
    "Age Group",
    "Number of Dependents",
    "Employment Duration",
    "Monthly Income",
    "Monthly Loan Repayment",
    "Requested Loan Amount",
]

ORDINAL_CATEGORIES: List[List[str]] = [
    # Age Group (6 tiers)
    ["Below 25", "25–34", "35–44", "45–54", "55–64", "65 or above"],
    # Number of Dependents (6 tiers)
    ["0", "1", "2", "3", "4", "5 or more"],
    # Employment Duration (6 tiers)
    [
        "Not employed/business",
        "Less than 1 year",
        "1–2 years",
        "3–5 years",
        "6–10 years",
        "More than 10 years",
    ],
    # Monthly Income (8 tiers)
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
    # Monthly Loan Repayment (7 tiers)
    [
        "No existing loan",
        "Below Rs.10,000",
        "Rs.10,000–24,999",
        "Rs.25,000–49,999",
        "Rs.50,000–74,999",
        "Rs.75,000–99,999",
        "Rs.100,000+",
    ],
    # Requested Loan Amount (9 tiers)
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

# Nominal Features to be One-Hot Encoded
NOMINAL_FEATURES: List[str] = [
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

TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------

def resolve_paths() -> Tuple[Path, Path, Path]:
    """Resolve absolute paths for input dataset, output processed directory, and evaluation directory."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent if script_dir.name == "data" else script_dir

    input_path = workspace_root / "data" / "synthetic" / DATASET_NAME
    output_dir = workspace_root / "data" / "processed"
    eval_dir = workspace_root / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return input_path, output_dir, eval_dir


# -----------------------------------------------------------------------------
# Preprocessing Pipeline Builder
# -----------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer combining:
    - OrdinalEncoder for hierarchically ordered categorical features.
    - OneHotEncoder for nominal categorical features (sparse_output=False for full interpretability).
    """
    ordinal_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    nominal_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_transformer, ORDINAL_FEATURES),
            ("nominal", nominal_transformer, NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# -----------------------------------------------------------------------------
# Main Preprocessing Execution Workflow
# -----------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute complete end-to-end dataset preprocessing pipeline for Dataset v2."""
    print("=" * 85)
    print("REVISED STEP 8: FINAL DATASET PREPROCESSING (GAUSSIAN COPULA DATASET v2)")
    print("=" * 85)

    input_path, output_dir, eval_dir = resolve_paths()
    print(f"Input Dataset Path : {input_path}")
    print(f"Output Directory   : {output_dir}")

    # 1. Load Dataset
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")

    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="cp1252")

    original_shape = df.shape
    print(f"\n[1] Primary Research Dataset Loaded: {original_shape[0]} rows, {original_shape[1]} columns")

    # 2. Validation Checks
    if original_shape != (3000, 17):
        raise ValueError(f"Expected shape (3000, 17), found: {original_shape}")

    missing_cols = [c for c in EXPECTED_PREDICTORS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    print(f"[2] Data Integrity Checks:")
    print(f"    - Missing Values : {missing_count}")
    print(f"    - Duplicate Rows : {duplicate_count}")

    if missing_count > 0:
        raise ValueError(f"Dataset contains {missing_count} missing values!")
    if duplicate_count > 0:
        raise ValueError(f"Dataset contains {duplicate_count} duplicate rows!")
    print("    [OK] Validation passed: 3,000 rows, 17 columns, 0 nulls, 0 duplicates.")

    # 3. Separate X and y
    X = df[EXPECTED_PREDICTORS].copy()
    y_raw = df[TARGET_COLUMN].copy()

    # 4. Target Encoding
    print("\n[3] Target Variable Encoding (Approved=1, Rejected=0):")
    app_total = int((y_raw == "Approved").sum())
    rej_total = int((y_raw == "Rejected").sum())
    print(f"    - Approved (1): {app_total} ({app_total / len(y_raw) * 100:.2f}%)")
    print(f"    - Rejected (0): {rej_total} ({rej_total / len(y_raw) * 100:.2f}%)")

    y = y_raw.map(TARGET_MAPPING)
    if y.isnull().any():
        raise ValueError(f"Encountered unmapped target labels in: {y_raw[y.isnull()].unique()}")

    # 5. Stratified Train / Test Split (BEFORE fitting)
    print(f"\n[4] Performing Stratified Train/Test Split (test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tr_app = int((y_train == 1).sum())
    tr_rej = int((y_train == 0).sum())
    te_app = int((y_test == 1).sum())
    te_rej = int((y_test == 0).sum())

    print(f"    - Training Set (X_train) : {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"    - Test Set (X_test)       : {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"    - Training Target Balance : Approved(1)={tr_app} ({tr_app/len(y_train)*100:.2f}%), Rejected(0)={tr_rej} ({tr_rej/len(y_train)*100:.2f}%)")
    print(f"    - Test Target Balance     : Approved(1)={te_app} ({te_app/len(y_test)*100:.2f}%), Rejected(0)={te_rej} ({te_rej/len(y_test)*100:.2f}%)")

    # 6. Fit Preprocessor ONLY on Training Data (Prevent Data Leakage)
    print("\n[5] Fitting Preprocessor Pipeline strictly on X_train...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # 7. Transform Datasets
    print("[6] Transforming X_train and X_test using fitted preprocessor...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract final feature names
    feature_names: List[str] = list(preprocessor.get_feature_names_out())
    num_features = len(feature_names)
    print(f"    - Total Encoded Features: {num_features}")
    print(f"      * Ordinal Features    : {len(ORDINAL_FEATURES)}")
    print(f"      * One-Hot Features    : {num_features - len(ORDINAL_FEATURES)}")

    # 8. Construct Transformed DataFrames
    df_X_train = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    df_X_test = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    df_y_train = pd.DataFrame(y_train, columns=[TARGET_COLUMN], index=y_train.index)
    df_y_test = pd.DataFrame(y_test, columns=[TARGET_COLUMN], index=y_test.index)

    # 9. Save Processed Artifacts
    print("\n[7] Saving Processed Data & Artifacts to 'data/processed/'...")
    x_train_file = output_dir / "X_train.csv"
    x_test_file = output_dir / "X_test.csv"
    y_train_file = output_dir / "y_train.csv"
    y_test_file = output_dir / "y_test.csv"
    preprocessor_file = output_dir / "preprocessor.joblib"
    feature_names_file = output_dir / "feature_names.json"

    df_X_train.to_csv(x_train_file, index=False, encoding="utf-8")
    df_X_test.to_csv(x_test_file, index=False, encoding="utf-8")
    df_y_train.to_csv(y_train_file, index=False, encoding="utf-8")
    df_y_test.to_csv(y_test_file, index=False, encoding="utf-8")
    joblib.dump(preprocessor, preprocessor_file)

    with open(feature_names_file, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    print(f"    [OK] Saved: {x_train_file.name} ({df_X_train.shape})")
    print(f"    [OK] Saved: {x_test_file.name} ({df_X_test.shape})")
    print(f"    [OK] Saved: {y_train_file.name} ({df_y_train.shape})")
    print(f"    [OK] Saved: {y_test_file.name} ({df_y_test.shape})")
    print(f"    [OK] Saved: {preprocessor_file.name}")
    print(f"    [OK] Saved: {feature_names_file.name}")

    # 10. Save Summary JSON (Task 5)
    summary_data = {
        "dataset_name": DATASET_NAME,
        "selected_synthesizer": SELECTED_SYNTHESIZER,
        "total_records": int(len(df)),
        "empirical_seed_records": 500,
        "synthetic_records": 2500,
        "total_raw_columns": int(df.shape[1]),
        "num_raw_predictors": len(EXPECTED_PREDICTORS),
        "target_distribution_overall": {
            "Approved": app_total,
            "Approved_percentage": round(app_total / len(df) * 100, 2),
            "Rejected": rej_total,
            "Rejected_percentage": round(rej_total / len(df) * 100, 2),
        },
        "train_test_split": {
            "train_samples": int(len(df_X_train)),
            "test_samples": int(len(df_X_test)),
            "split_ratio": "80/20",
            "stratified": True,
            "random_state": RANDOM_STATE,
            "train_target_distribution": {
                "Approved": tr_app,
                "Approved_percentage": round(tr_app / len(df_X_train) * 100, 2),
                "Rejected": tr_rej,
                "Rejected_percentage": round(tr_rej / len(df_X_train) * 100, 2),
            },
            "test_target_distribution": {
                "Approved": te_app,
                "Approved_percentage": round(te_app / len(df_X_test) * 100, 2),
                "Rejected": te_rej,
                "Rejected_percentage": round(te_rej / len(df_X_test) * 100, 2),
            },
        },
        "transformed_feature_count": num_features,
        "preprocessing_configuration": {
            "ordinal_features": ORDINAL_FEATURES,
            "ordinal_count": len(ORDINAL_FEATURES),
            "nominal_features": NOMINAL_FEATURES,
            "nominal_count": len(NOMINAL_FEATURES),
            "one_hot_encoded_feature_count": num_features - len(ORDINAL_FEATURES),
            "leakage_prevention": "ColumnTransformer fitted strictly on X_train only",
        },
    }

    summary_file = eval_dir / "final_dataset_v2_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"    [OK] Saved: {summary_file.name}")

    # 11. Verification of Transformed Matrices
    assert df_X_train.shape == (2400, 55), f"Unexpected X_train shape: {df_X_train.shape}"
    assert df_X_test.shape == (600, 55), f"Unexpected X_test shape: {df_X_test.shape}"
    assert df_y_train.shape == (2400, 1), f"Unexpected y_train shape: {df_y_train.shape}"
    assert df_y_test.shape == (600, 1), f"Unexpected y_test shape: {df_y_test.shape}"
    assert df_X_train.isnull().sum().sum() == 0, "Null values detected in X_train!"
    assert df_X_test.isnull().sum().sum() == 0, "Null values detected in X_test!"

    print("\n" + "=" * 85)
    print("[SUCCESS] REVISED STEP 8 PREPROCESSING COMPLETED PERFECTLY")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
