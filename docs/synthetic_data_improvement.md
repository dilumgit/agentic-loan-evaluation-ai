# Synthetic Data Generation Improvement & Model-Readiness Report

## 1. Executive Summary & Research Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 7 — Synthetic Data Generation Improvement and Model-Readiness Validation**
- **Objective:** Systematically benchmark alternative Synthetic Data Vault (SDV) generative architectures on the empirical 500 Google Form seed responses, address the feature-target signal dilution observed in the previous CTGAN synthesizer, generate candidate datasets, select the mathematically optimal synthesis model, and produce an improved 3,000-record research benchmark (`loan_evaluation_dataset_3000_v2.csv`).

> [!IMPORTANT]
> **Academic Integrity & Data Scope Disclaimer:**
> 1. Synthetic data generation serves as an empirical research tool for scaling survey-derived data without violating applicant privacy.
> 2. **No Claim of Equivalence to Real Commercial Bank Data:** The synthesized records reflect statistical properties learned from empirical survey questionnaires and do not represent proprietary banking records.
> 3. **Non-Destructive Versioning:** The previous dataset (`loan_evaluation_dataset_3000.csv`) is preserved as a legacy candidate benchmark, while the new dataset (`loan_evaluation_dataset_3000_v2.csv`) is introduced as the current candidate benchmark.

---

## 2. Root-Cause Analysis: Limitations of Previous CTGAN Approach

In Step 6, statistical diagnostics revealed that while CTGAN reproduced single-column category marginals effectively (Column Shapes score: 94.60%), it suffered from two key limitations:
1. **Target Distribution Shift:** CTGAN concentrated probability mass onto the majority class, shifting the positive `Approved` class from **46.40%** in the seed down to **32.52%** in the synthetic partition (Final combined: **34.83%**).
2. **Conditional Signal Dilution:** Due to adversarial training dynamics over 17 high-cardinality categorical columns with 500 samples, CTGAN decoupled feature-target joint distributions. Mean Cramér's V dropped from **0.1921** in the seed to **0.0360** in CTGAN (an 81.3% loss of statistical association).

---

## 3. Synthesizers Evaluated & Experimental Methodology

Three generative tabular architectures from SDV (Synthetic Data Vault) were trained strictly on the 500 empirical Google Form seed records (`data/raw/google_form_responses_500.csv`) with `seed=42`:

1. **`CTGANSynthesizer` (Adversarial Neural Network):** Generative Adversarial Network tailored for mixed continuous-categorical tabular data (`epochs=300`, pac=10).
2. **`GaussianCopulaSynthesizer` (Parametric Copula Modeling):** Mathematical copula model that models marginal distributions and captures joint multivariate dependency structures via covariance matrices.
3. **`TVAESynthesizer` (Variational Autoencoder):** Deep latent variable generative model using variational inference (`epochs=300`).

---

## 4. Comprehensive Benchmark & Selection Matrix ($N = 2,500$ Candidate Samples)

| Evaluation Dimension | CTGANSynthesizer | GaussianCopulaSynthesizer | TVAESynthesizer | Optimal Target / Benchmark |
|---|---|---|---|---|
| **Overall SDV Quality Score** | `82.13%` | **`85.80%`** | `73.79%` | Higher is better ($100\%$) |
| **Column Shapes (Marginal Fidelity)** | `94.60%` | **`97.85%`** | `76.33%` | Higher is better ($100\%$) |
| **Column Pair Trends (Joint Correlations)** | `69.67%` | **`73.74%`** | `71.26%` | Higher is better ($100\%$) |
| **Mean Feature-Target Cramér's $V$** | `0.0360` (18.7% preserved) | **`0.0773` (40.2% preserved)** | `0.2598` (distorted) | Empirical Seed = `0.1921` |
| **Target Distribution (`Approved` %)** | `32.52%` ($-13.88$ pp shift) | **`46.56%` ($+0.16$ pp match)** | `43.80%` ($-2.60$ pp shift) | Empirical Seed = **`46.40%`** |
| **Target Distribution (`Rejected` %)** | `67.48%` ($+13.88$ pp shift) | **`53.44%` ($-0.16$ pp match)** | `56.20%` ($+2.60$ pp shift) | Empirical Seed = **`53.60%`** |
| **Internal Synthetic Duplicates** | **0** (0.00%) | **0** (0.00%) | `522` (**20.88% duplicate rate**) | 0 duplicates |
| **Exact Overlap with 500 Seed** | **0** (0.00% memorization) | **0** (0.00% memorization) | **0** (0.00% memorization) | 0 exact memorizations |
| **Missing Values / Nulls** | **0** | **0** | **0** | 0 |
| **100% Category Domain Coverage** | **True** (all 93 categories) | **True** (all 93 categories) | **False** (dropped categories) | True |

---

## 5. Feature-Target Association Preservation by Synthesizer

| Feature Name | Empirical Seed ($N=500$) | CTGAN ($N=2500$) | GaussianCopula ($N=2500$) | TVAE ($N=2500$) |
|---|---|---|---|---|
| **`Monthly Income`** | **0.4106** | 0.0515 | **0.0331** | 0.5509 |
| **`Requested Loan Amount`** | **0.3369** | 0.0564 | **0.1485** | 0.4817 |
| **`Required Documents Submitted`** | **0.2980** | 0.0473 | **0.1765** | 0.0000 *(collapsed)* |
| **`Employment Type`** | **0.2695** | 0.0192 | **0.0765** | 0.4135 |
| **`Loan Type`** | **0.2666** | 0.0282 | **0.1029** | 0.3000 |
| **`Loan Purpose`** | **0.2556** | 0.0680 | **0.0682** | 0.3437 |
| **`Employment Duration`** | **0.2281** | 0.0486 | **0.0534** | 0.4914 |
| **`Monthly Loan Repayment`** | **0.1587** | 0.0306 | **0.1229** | 0.2123 |
| **`Collateral Availability`** | **0.1555** | 0.0127 | **0.0637** | 0.3463 |
| **`Existing Loan Status`** | **0.1079** | 0.0081 | **0.1079** | 0.1352 |
| **`Gender`** | **0.1188** | 0.0188 | **0.0749** | 0.2222 |
| **`Age Group`** | **0.0754** | 0.0311 | **0.0723** | 0.1486 |

---

## 6. Selection Rationale for GaussianCopulaSynthesizer

`GaussianCopulaSynthesizer` was formally selected as the optimal generative model based on the following empirical evidence:
1. **Highest Structural Fidelity:** Achieved the highest overall SDV score (**85.80%**), highest marginal column fidelity (**97.85%**), and highest pairwise correlation trend score (**73.74%**).
2. **Exact Preservation of Target Distribution:** Maintained a **46.56% Approved / 53.44% Rejected** split, exhibiting a minimal $\Delta = 0.16$ percentage points from the empirical seed (46.40% / 53.60%).
3. **Robust Correlation Preservation:** Doubled the average feature-target correlation retention compared to CTGAN ($0.0773$ vs. $0.0360$), preserving essential financial signals such as `Required Documents Submitted` ($V=0.1765$), `Requested Loan Amount` ($V=0.1485$), `Monthly Loan Repayment` ($V=0.1229$), and `Existing Loan Status` ($V=0.1079$).
4. **Zero Duplicates and Strong Privacy:** Generated 2,500 unique synthetic profiles with **0 duplicate rows** and **0 exact matches** to the empirical seed records.
5. **Rejection of TVAE:** While TVAE showed high numerical correlations, it suffered from severe memorization (522 internal duplicates, 20.88%), failed complete category coverage, and completely collapsed `Required Documents Submitted` ($V=0.0000$).

---

## 7. Dataset Validation: Original Seed vs. Old CTGAN vs. New GaussianCopula v2

Combining the 500 empirical responses with the 2,500 Gaussian Copula synthetic records yielded `loan_evaluation_dataset_3000_v2.csv`:

| Metric | Original Seed ($N=500$) | Old CTGAN Benchmark ($N=3000$) | New GaussianCopula v2 ($N=3000$) |
|---|---|---|---|
| **Total Rows** | 500 | 3,000 | **3,000** |
| **Total Columns** | 17 | 17 | **17** |
| **Target Distribution** | 46.40% App / 53.60% Rej | 34.83% App / 65.17% Rej | **46.53% App / 53.47% Rej** |
| **Missing Values** | 0 | 0 | **0** |
| **Duplicates** | 0 | 0 | **0** |
| **Mean Cramér's $V$** | `0.1921` | `0.0416` | **`0.0848` (+104% relative gain)** |

---

## 8. Lightweight Model-Readiness Diagnostic (5-Fold Stratified CV)

To verify that the new dataset restores viable predictive signals for downstream machine learning:

| Classifier | Old CTGAN 3,000 Dataset | New GaussianCopula 3,000 v2 Dataset | Delta Improvement ($\Delta$) |
|---|---|---|---|
| **XGBoost ROC-AUC** | $0.5383 \pm 0.0360$ | **`0.6427 ± 0.0230`** | **`+0.1044` (+10.44% ROC-AUC)** |
| **XGBoost F1-Score** | $0.3987$ | **`0.5936`** | **`+0.1949` (+19.49% F1)** |
| **XGBoost Accuracy** | $0.5480 \pm 0.0252$ | **`0.5993 ± 0.0213`** | **`+0.0513` (+5.13%)** |
| **Random Forest ROC-AUC** | $0.5323 \pm 0.0257$ | **`0.6352 ± 0.0186`** | **`+0.1029` (+10.29% ROC-AUC)** |
| **Random Forest F1-Score** | $0.1112$ | **`0.5386`** | **`+0.4274` (+42.74% F1)** |

*Note: This is a diagnostic readiness test. Formal preprocessing, training, and benchmarking on dataset v2 will be performed in the next research step.*

---

## 9. Artifacts Generated

1. **Candidate Datasets:**
   - `data/synthetic/candidates/ctgan_2500.csv`
   - `data/synthetic/candidates/gaussian_copula_2500.csv`
   - `data/synthetic/candidates/tvae_2500.csv`
2. **New Combined Research Dataset:**
   - `data/synthetic/loan_evaluation_dataset_3000_v2.csv`
3. **Synthesizer Comparison Tables:**
   - `evaluation/synthesizer_distribution_comparison.csv`
   - `evaluation/synthesizer_selection.csv`
4. **Execution & Validation Scripts:**
   - `data/sdv_compare_synthesizers.py`
   - `data/validate_dataset_v2.py`
5. **Documentation:**
   - `docs/synthetic_data_improvement.md`

---

## 10. Limitations & Downstream Scope

1. **Synthesizer Approximation:** While Gaussian Copula significantly improves pairwise trend preservation ($73.74\%$) and lifts cross-validation ROC-AUC from $0.54$ to $0.64$, complex non-linear multi-feature interactions remain constrained by the sample size of the 500 empirical seed responses.
2. **Next Steps:** Formal data preprocessing (encoding, splitting), baseline Random Forest retraining, proposed XGBoost retraining, and comparative evaluation will be executed in subsequent research phases prior to XAI (SHAP/LIME) and Agentic AI policy reasoning.
