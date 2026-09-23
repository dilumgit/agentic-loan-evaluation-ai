"""
Dataset Inspection Script for Loan Evaluation Academic Research Project.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

import io
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure utf-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def find_dataset_file() -> Path:
    """Locate the synthetic dataset file across common relative paths."""
    filename = "loan_evaluation_synthetic_2000_up_to_100m.csv"
    
    # Potential locations relative to current working dir and script dir
    candidate_paths = [
        Path("data") / "synthetic" / filename,
        Path("synthetic") / filename,
        Path(__file__).resolve().parent / "synthetic" / filename,
        Path(__file__).resolve().parent.parent / "data" / "synthetic" / filename,
        Path(filename),
    ]
    
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.resolve()
            
    return candidate_paths[0].resolve()


def inspect_dataset(file_path: Path) -> None:
    """Inspect dataset properties and print comprehensive statistics."""
    print("=" * 80)
    print("LOAN EVALUATION DATASET INSPECTION REPORT")
    print("=" * 80)
    print(f"Target File: {file_path}")
    
    if not file_path.exists():
        print("\n[ERROR] Dataset file not found!")
        print(f"Please place '{file_path.name}' inside 'data/synthetic/'")
        return

    # 1. Load Dataset
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"\n[ERROR] Failed to read CSV file: {e}")
        return

    # 2. Shape
    rows, cols = df.shape
    print("\n--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Records) : {rows}")
    print(f"Total Columns (Features): {cols}")

    # 3. Column Names & Data Types
    print("\n--- 2. COLUMNS & DATA TYPES ---")
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean() * 100).round(2).values
    })
    print(col_info.to_string(index=False))

    # 4. Missing Values Summary
    total_missing = int(df.isnull().sum().sum())
    print("\n--- 3. MISSING VALUES SUMMARY ---")
    print(f"Total Missing Values in Dataset: {total_missing}")
    if total_missing > 0:
        missing_series = df.isnull().sum()
        print(missing_series[missing_series > 0].to_string())
    else:
        print("No missing values detected across all columns.")

    # 5. Duplicate Rows
    duplicate_count = int(df.duplicated().sum())
    print("\n--- 4. DUPLICATE ROWS ---")
    print(f"Total Duplicate Rows: {duplicate_count}")

    # 6. Target Class Distribution
    target_col = None
    possible_targets = ["Loan Application Result", "loan_application_result", "Target", "target"]
    for col in df.columns:
        if col.strip().lower() in [t.lower() for t in possible_targets]:
            target_col = col
            break

    print("\n--- 5. TARGET VARIABLE ANALYSIS ---")
    if target_col:
        print(f"Identified Target Column: '{target_col}'")
        target_counts = df[target_col].value_counts(dropna=False)
        target_percentages = df[target_col].value_counts(normalize=True, dropna=False) * 100
        target_dist = pd.DataFrame({
            "Count": target_counts,
            "Percentage (%)": target_percentages.round(2)
        })
        print(target_dist.to_string())
    else:
        print("[WARNING] Target column 'Loan Application Result' not found directly in dataset.")

    # 7. Unique Values for Categorical Columns
    print("\n--- 6. CATEGORICAL FEATURES & UNIQUE VALUES ---")
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            print(f"\n* Feature: '{col}' (Unique count: {len(unique_vals)})")
            if len(unique_vals) <= 15:
                for val in unique_vals:
                    count = (df[col] == val).sum()
                    pct = (count / rows) * 100
                    print(f"   - {val}: {count} ({pct:.2f}%)")
            else:
                print(f"   - Sample values: {list(unique_vals[:10])} ... (+{len(unique_vals) - 10} more)")
    else:
        print("No categorical/object columns found.")

    # 8. Descriptive Statistics for Numeric Columns
    print("\n--- 7. NUMERICAL FEATURES DESCRIPTIVE STATISTICS ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().round(2).to_string())
    else:
        print("No purely numeric columns detected (all features are categorical/binned strings).")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    dataset_path = find_dataset_file()
    inspect_dataset(dataset_path)
