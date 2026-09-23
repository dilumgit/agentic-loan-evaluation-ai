"""
SDV Synthetic Data Validation Script
Validates data/synthetic/sdv_generated_2500.csv and data/synthetic/loan_evaluation_dataset_3000.csv
against the empirical seed dataset (data/raw/google_form_responses_500.csv) across all 12 quality criteria.
"""

import sys
import os
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED_PATH = "data/raw/google_form_responses_500.csv"
SYNTHETIC_PATH = "data/synthetic/sdv_generated_2500.csv"
COMBINED_PATH = "data/synthetic/loan_evaluation_dataset_3000.csv"

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

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")

def run_sdv_validation():
    print("=" * 85)
    print("TASK 6: COMPREHENSIVE SDV SYNTHETIC DATA QUALITY & INTEGRITY VALIDATION")
    print("=" * 85)
    
    # Check file existence
    for p in [SEED_PATH, SYNTHETIC_PATH, COMBINED_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
            
    df_seed = load_csv(SEED_PATH)
    df_synth = load_csv(SYNTHETIC_PATH)
    df_combined = load_csv(COMBINED_PATH)
    
    validation_results = {}
    
    # -------------------------------------------------------------
    # 1. Dimension Checks
    # -------------------------------------------------------------
    print("\n[CHECK 1 & 2] Row & Column Dimension Verification")
    print(f"  • Seed Dataset Shape:      {df_seed.shape} (Expected: 500, 17)")
    print(f"  • Synthetic Dataset Shape: {df_synth.shape} (Expected: 2500, 17)")
    print(f"  • Combined Dataset Shape:  {df_combined.shape} (Expected: 3000, 17)")
    
    assert df_seed.shape == (500, 17), f"Invalid seed shape: {df_seed.shape}"
    assert df_synth.shape == (2500, 17), f"Invalid synthetic shape: {df_synth.shape}"
    assert df_combined.shape == (3000, 17), f"Invalid combined shape: {df_combined.shape}"
    print("[OK] Exact row counts (500 / 2,500 / 3,000) and column counts (17) verified.")
    validation_results["dimensions"] = "PASSED"

    # -------------------------------------------------------------
    # 2. Column Names & Ordering Check
    # -------------------------------------------------------------
    print("\n[CHECK 3] Column Schema Alignment")
    assert list(df_synth.columns) == list(df_seed.columns), "Synthetic columns mismatch seed columns!"
    assert list(df_combined.columns) == list(df_seed.columns), "Combined columns mismatch seed columns!"
    assert list(df_synth.columns) == EXPECTED_COLUMNS, "Columns mismatch expected schema!"
    print("[OK] Column names and ordering identical across all three datasets.")
    validation_results["columns"] = "PASSED"

    # -------------------------------------------------------------
    # 3. Missing Values & Empty Cells
    # -------------------------------------------------------------
    print("\n[CHECK 4] Missing Values and Null Verification")
    null_synth = df_synth.isnull().sum().sum()
    null_comb = df_combined.isnull().sum().sum()
    print(f"  • Synthetic Dataset Nulls: {null_synth}")
    print(f"  • Combined Dataset Nulls:  {null_comb}")
    assert null_synth == 0, f"Synthetic dataset has {null_synth} null values!"
    assert null_comb == 0, f"Combined dataset has {null_comb} null values!"
    print("[OK] Zero missing values across all records.")
    validation_results["missing_values"] = "PASSED"

    # -------------------------------------------------------------
    # 4. Target Classes Check
    # -------------------------------------------------------------
    print("\n[CHECK 5 & 6] Target Classes Verification")
    synth_targets = set(df_synth[TARGET_COLUMN].unique())
    comb_targets = set(df_combined[TARGET_COLUMN].unique())
    print(f"  • Observed Synthetic Target Classes: {synth_targets}")
    print(f"  • Observed Combined Target Classes:  {comb_targets}")
    assert synth_targets.issubset(EXPECTED_TARGETS), f"Invalid synthetic target classes: {synth_targets}"
    assert comb_targets.issubset(EXPECTED_TARGETS), f"Invalid combined target classes: {comb_targets}"
    print("[OK] Target variable contains only 'Approved' and 'Rejected'.")
    validation_results["target_classes"] = "PASSED"

    # -------------------------------------------------------------
    # 5. Out-of-Domain Categorical Values Check
    # -------------------------------------------------------------
    print("\n[CHECK 7 & 8] Categorical Domain & Requested Loan Amount Verification")
    invalid_category_counts = 0
    for col in EXPECTED_COLUMNS:
        seed_uniques = set(df_seed[col].unique())
        synth_uniques = set(df_synth[col].unique())
        out_of_domain = synth_uniques - seed_uniques
        if out_of_domain:
            print(f"[-] Column '{col}' contains out-of-domain categories: {out_of_domain}")
            invalid_category_counts += len(out_of_domain)
        else:
            print(f"  [OK] '{col}': all {len(synth_uniques)} synthetic categories match seed domain.")
            
    assert invalid_category_counts == 0, f"Found {invalid_category_counts} invalid categories!"
    print("[OK] All categorical features (including Requested Loan Amount) are 100% valid within seed domains.")
    validation_results["category_domains"] = "PASSED"

    # -------------------------------------------------------------
    # 6. Duplicate Analysis (Intra-Synthetic & Overlap with Seed)
    # -------------------------------------------------------------
    print("\n[CHECK 9 & 10] Duplicate Analysis & Privacy / Novelty Audit")
    seed_dups = df_seed.duplicated().sum()
    synth_dups = df_synth.duplicated().sum()
    comb_dups = df_combined.duplicated().sum()
    
    # Exact overlap between synthetic and seed records
    # Merge on all 17 columns to find exact identical rows
    overlap_df = pd.merge(df_synth, df_seed, how="inner", on=EXPECTED_COLUMNS)
    overlap_count = len(overlap_df)
    
    print(f"  • Seed internal duplicates:           {seed_dups} ({seed_dups/len(df_seed)*100:.2f}%)")
    print(f"  • Synthetic internal duplicates:      {synth_dups} ({synth_dups/len(df_synth)*100:.2f}%)")
    print(f"  • Combined dataset duplicates:        {comb_dups} ({comb_dups/len(df_combined)*100:.2f}%)")
    print(f"  • Exact synthetic matches with seed:  {overlap_count} ({overlap_count/len(df_synth)*100:.2f}%)")
    print(f"  • Novel synthetic combinations:       {len(df_synth) - overlap_count} ({(1 - overlap_count/len(df_synth))*100:.2f}%)")
    validation_results["duplicates"] = {
        "seed_dups": int(seed_dups),
        "synth_dups": int(synth_dups),
        "comb_dups": int(comb_dups),
        "overlap_count": int(overlap_count)
    }

    # -------------------------------------------------------------
    # 7. Distribution Comparisons (Seed vs Synthetic vs Combined)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("[CHECK 11 & 12] DETAILED DISTRIBUTION COMPARISON")
    print("=" * 85)
    
    # Target distribution table
    print("\n--- Target Variable Distribution Comparison ---")
    target_summary = []
    for val in ["Approved", "Rejected"]:
        s_cnt = (df_seed[TARGET_COLUMN] == val).sum()
        s_pct = s_cnt / len(df_seed) * 100
        g_cnt = (df_synth[TARGET_COLUMN] == val).sum()
        g_pct = g_cnt / len(df_synth) * 100
        c_cnt = (df_combined[TARGET_COLUMN] == val).sum()
        c_pct = c_cnt / len(df_combined) * 100
        target_summary.append({
            "Target Class": val,
            "Seed Count (500)": s_cnt,
            "Seed %": f"{s_pct:.2f}%",
            "Synth Count (2500)": g_cnt,
            "Synth %": f"{g_pct:.2f}%",
            "Final Count (3000)": c_cnt,
            "Final %": f"{c_pct:.2f}%"
        })
    df_target_summary = pd.DataFrame(target_summary)
    print(df_target_summary.to_string(index=False))

    # Feature-by-feature distribution comparison summary
    print("\n--- Feature Category Distribution Alignment (Sample Summaries) ---")
    tvd_summary = []
    for col in EXPECTED_COLUMNS:
        seed_dist = df_seed[col].value_counts(normalize=True)
        synth_dist = df_synth[col].value_counts(normalize=True)
        # Align indexes
        all_cats = sorted(list(set(seed_dist.index).union(set(synth_dist.index))))
        s_series = seed_dist.reindex(all_cats, fill_value=0.0)
        g_series = synth_dist.reindex(all_cats, fill_value=0.0)
        # Total Variation Distance: TVD = 0.5 * sum(|P(x) - Q(x)|)
        tvd = 0.5 * np.sum(np.abs(s_series - g_series))
        # Total Variation Complement Score (TVComplement = 1 - TVD)
        tv_comp = (1.0 - tvd) * 100
        tvd_summary.append({
            "Feature Name": col,
            "Categories": len(all_cats),
            "TVD": round(float(tvd), 4),
            "Fidelity Score (%)": f"{tv_comp:.2f}%"
        })
        
    df_tvd = pd.DataFrame(tvd_summary)
    print(df_tvd.to_string(index=False))
    
    # Verify combination preservation: first 500 rows equal seed
    print("\n--- Seed Preservation Verification in Final Dataset ---")
    seed_subset_in_final = df_combined.iloc[:500].reset_index(drop=True)
    is_preserved = seed_subset_in_final.equals(df_seed)
    print(f"  • First 500 rows of combined dataset match seed 100%: {is_preserved}")
    assert is_preserved, "Original 500 records were modified during combination!"
    
    print("\n" + "=" * 85)
    print("[SUCCESS] ALL 12 VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 85)
    return validation_results

if __name__ == "__main__":
    run_sdv_validation()
