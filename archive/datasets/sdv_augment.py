"""
SDV Synthetic Data Augmentation Pipeline
Step 2.5: Generates 2,500 synthetic records from the 500-record empirical Google Form seed dataset
using the official SDV CTGAN Synthesizer, yielding a final combined research dataset of 3,000 records.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
EPOCHS = 300
SYNTHETIC_ROWS = 2500

SEED_DATA_PATH = "data/raw/google_form_responses_500.csv"
SYNTHETIC_OUT_PATH = "data/synthetic/sdv_generated_2500.csv"
COMBINED_OUT_PATH = "data/synthetic/loan_evaluation_dataset_3000.csv"
MODEL_OUT_PATH = "data/synthetic/sdv_model.pkl"
METADATA_OUT_PATH = "data/synthetic/sdv_metadata.json"

def set_deterministic_seed(seed: int = SEED):
    """Sets random seeds across Python, NumPy, and PyTorch for 100% reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Configure deterministic algorithms in PyTorch if available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_sdv_augmentation():
    print("=" * 80)
    print("STEP 2.5: SDV SYNTHETIC DATA AUGMENTATION (CTGAN SYNTHESIZER)")
    print("=" * 80)
    
    set_deterministic_seed(SEED)
    print(f"[+] Deterministic random seed set to: {SEED}")
    
    # 1. Load empirical seed dataset
    if not os.path.exists(SEED_DATA_PATH):
        raise FileNotFoundError(f"Seed dataset not found at: {SEED_DATA_PATH}")
    
    try:
        df_seed = pd.read_csv(SEED_DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df_seed = pd.read_csv(SEED_DATA_PATH, encoding="cp1252")
        
    print(f"[+] Loaded seed dataset from: {SEED_DATA_PATH}")
    print(f"    - Shape: {df_seed.shape} (Records: {len(df_seed)}, Columns: {len(df_seed.columns)})")
    print(f"    - Target distribution:\n{df_seed['Loan Application Result'].value_counts().to_string()}")
    
    if len(df_seed) != 500 or len(df_seed.columns) != 17:
        raise ValueError(f"Seed dataset must be exactly 500 rows and 17 columns. Found: {df_seed.shape}")
        
    # 2. Configure Single Table Metadata
    print("\n[+] Detecting single-table metadata schema from seed dataset...")
    metadata = Metadata.detect_from_dataframe(data=df_seed, table_name="loan_evaluation")
    
    os.makedirs("data/synthetic", exist_ok=True)
    metadata.save_to_json(filepath=METADATA_OUT_PATH)
    print(f"[OK] Saved SDV metadata specification to: {METADATA_OUT_PATH}")
    
    # 3. Initialize and train CTGAN Synthesizer
    print(f"\n[+] Initializing CTGANSynthesizer (epochs={EPOCHS}, seed={SEED})...")
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=EPOCHS,
        verbose=False
    )
    
    print("[+] Fitting CTGAN Synthesizer on the 500 empirical records...")
    synthesizer.fit(df_seed)
    print("[OK] CTGAN Synthesizer training complete.")
    
    # 4. Save the fitted model
    synthesizer.save(filepath=MODEL_OUT_PATH)
    print(f"[OK] Fitted synthesizer saved to: {MODEL_OUT_PATH}")
    
    # 5. Generate exactly 2,500 synthetic records
    print(f"\n[+] Sampling exactly {SYNTHETIC_ROWS} synthetic records...")
    df_synthetic = synthesizer.sample(num_rows=SYNTHETIC_ROWS)
    
    if len(df_synthetic) != SYNTHETIC_ROWS or len(df_synthetic.columns) != len(df_seed.columns):
        raise ValueError(f"Synthetic generation produced invalid dimensions: {df_synthetic.shape}")
        
    # Ensure column order matches seed dataset exactly
    df_synthetic = df_synthetic[df_seed.columns]
    
    # Save standalone synthetic dataset
    df_synthetic.to_csv(SYNTHETIC_OUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] Generated synthetic dataset saved to: {SYNTHETIC_OUT_PATH}")
    print(f"    - Shape: {df_synthetic.shape}")
    print(f"    - Synthetic target distribution:\n{df_synthetic['Loan Application Result'].value_counts().to_string()}")
    
    # 6. Combine: 500 original records + 2,500 synthetic records = 3,000 total records
    print(f"\n[+] Combining 500 original records + 2,500 synthetic records into final dataset...")
    # First 500 records are the exact, unmodified seed records
    df_final = pd.concat([df_seed, df_synthetic], ignore_index=True)
    
    if len(df_final) != 3000 or len(df_final.columns) != 17:
        raise ValueError(f"Final dataset has invalid dimensions: {df_final.shape}")
        
    df_final.to_csv(COMBINED_OUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] Final 3,000-record research dataset saved to: {COMBINED_OUT_PATH}")
    print(f"    - Total Shape: {df_final.shape}")
    print(f"    - Original Records (rows 0-499): {len(df_seed)}")
    print(f"    - Synthetic Records (rows 500-2999): {len(df_synthetic)}")
    print(f"    - Final combined target distribution:\n{df_final['Loan Application Result'].value_counts().to_string()}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] SDV AUGMENTATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return df_seed, df_synthetic, df_final

if __name__ == "__main__":
    run_sdv_augmentation()
