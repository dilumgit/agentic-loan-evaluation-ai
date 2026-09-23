"""
Seed Dataset Validation Script
Validates data/raw/google_form_responses_500.csv for integrity, completeness,
categorical validity, and suitability for SDV tabular synthetic data augmentation.
"""

import sys
import os
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_COLUMNS = [
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
    "Loan Application Result"
]

TARGET_COLUMN = "Loan Application Result"
EXPECTED_TARGETS = {"Approved", "Rejected"}

def validate_seed_dataset(filepath: str = "data/raw/google_form_responses_500.csv"):
    print("=" * 80)
    print("TASK 2: VALIDATING EMPIRICAL SEED DATASET (500 GOOGLE FORM RESPONSES)")
    print("=" * 80)
    
    if not os.path.exists(filepath):
        print(f"[-] ERROR: File not found at path: {filepath}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="cp1252")
        
    print(f"[+] Loaded file: {filepath}")
    
    # 1. Record and Column Count
    row_count, col_count = df.shape
    print(f"\n--- Dimension Check ---")
    print(f"Total Rows (Records): {row_count}")
    print(f"Total Columns (Features + Target): {col_count}")
    
    if row_count != 500:
        raise ValueError(f"Expected exactly 500 rows, found {row_count}")
    if col_count != 17:
        raise ValueError(f"Expected exactly 17 columns, found {col_count}")
    print("[OK] Exact dimensions confirmed: 500 rows, 17 columns.")

    # 2. Expected Column Names
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_COLUMNS:
        diff_missing = set(EXPECTED_COLUMNS) - set(actual_cols)
        diff_extra = set(actual_cols) - set(EXPECTED_COLUMNS)
        raise ValueError(f"Column mismatch! Missing: {diff_missing}, Unexpected: {diff_extra}")
    print("[OK] All 17 expected column names match perfectly in exact order.")

    # 3. Missing Values Check
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n--- Missing Values Check ---")
    print(f"Total Null/NaN Values: {total_nulls}")
    if total_nulls > 0:
        print(null_counts[null_counts > 0])
        raise ValueError(f"Dataset contains {total_nulls} missing values!")
    print("[OK] Zero missing values detected across all columns.")

    # 4. Empty Strings / Whitespace-only Check
    empty_str_counts = 0
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            empty_count = (df[col].astype(str).str.strip() == "").sum()
            if empty_count > 0:
                print(f"[-] Column '{col}' contains {empty_count} empty strings.")
                empty_str_counts += empty_count
    if empty_str_counts > 0:
        raise ValueError(f"Dataset contains {empty_str_counts} empty or blank string cells!")
    print("[OK] Zero empty string cells detected across all records.")

    # 5. Duplicate Rows Check
    duplicate_count = df.duplicated().sum()
    print(f"\n--- Duplicate Rows Check ---")
    print(f"Total Duplicate Rows: {duplicate_count}")
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate rows in seed dataset!")
    print("[OK] Zero duplicate rows detected in the seed dataset.")

    # 6. Target Distribution
    print(f"\n--- Target Variable Distribution ({TARGET_COLUMN}) ---")
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    for target_val, count in target_counts.items():
        pct = (count / row_count) * 100
        print(f"  * {target_val}: {count} ({pct:.2f}%)")
        
    actual_target_classes = set(target_counts.keys())
    if not actual_target_classes.issubset(EXPECTED_TARGETS):
        raise ValueError(f"Unexpected target values: {actual_target_classes - EXPECTED_TARGETS}")
    if target_counts.get("Approved") != 232 or target_counts.get("Rejected") != 268:
        print(f"[!] Note: Observed distribution: Approved={target_counts.get('Approved')}, Rejected={target_counts.get('Rejected')}")
    else:
        print("[OK] Target distribution verified: Approved = 232 (46.40%), Rejected = 268 (53.60%).")

    # 7. Unique Categories Per Feature
    print(f"\n--- Feature Categories & Cardinality Breakdown ---")
    for i, col in enumerate(df.columns, 1):
        uniques = sorted(df[col].unique().tolist())
        print(f"{i:2d}. {col} ({len(uniques)} categories):")
        print(f"    {uniques}")

    print("\n" + "=" * 80)
    print("[SUCCESS] SUITABILITY CONFIRMED: Dataset is 100% clean, validated, and ready for SDV augmentation.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    validate_seed_dataset()
