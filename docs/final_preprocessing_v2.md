# Final Dataset Preprocessing Report (Dataset v2 — Gaussian Copula)

## 1. Executive Summary & Research Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 8 — Final Dataset Preprocessing for Selected Gaussian Copula Dataset (`loan_evaluation_dataset_3000_v2.csv`)**
- **Objective:** Establish the formal, leakage-free preprocessing pipeline for the improved 3,000-record research benchmark dataset (`loan_evaluation_dataset_3000_v2.csv`). This pipeline encodes raw survey attributes into numeric feature tensors while strictly isolating holdout test data to prevent data leakage.

> [!IMPORTANT]
> **Academic Integrity & Data Scope Disclaimer:**
> 1. The primary research dataset consists of **500 empirical questionnaire responses** collected via Google Form surveys combined with **2,500 Gaussian Copula-generated synthetic observations** based on the statistical properties of the 500 empirical responses.
> 2. The synthetic instances do not represent actual commercial bank customer records.
> 3. The previous 3,000-record dataset (`loan_evaluation_dataset_3000.csv`) is preserved as a **LEGACY CTGAN CANDIDATE** for historical comparison.

---

## 2. Why Gaussian Copula Dataset (v2) Was Selected
Following the multi-synthesizer benchmark in Step 7:
1. **Superior Correlation Retention:** Gaussian Copula achieved an average pairwise column trend score of **73.74%** and more than doubled the mean feature-target Cramér's $V$ ($0.0848$ vs. CTGAN $0.0416$).
2. **Exact Target Distribution Match:** Successfully maintained the empirical class distribution at **46.53% Approved / 53.47% Rejected** (matching the 500-record empirical seed: 46.40% / 53.60%, with a minimal $\Delta = 0.13\text{ pp}$). In contrast, CTGAN had suffered an undesirable $-11.57\text{ pp}$ target shift.
3. **Zero Memorization and Complete Domain Coverage:** Generated 2,500 unique synthetic observations with **0 internal duplicate rows**, **0 exact matches to seed records**, and complete coverage of all 93 category domain levels across 17 attributes.
4. **Demonstrated Predictive Lift:** In lightweight model-readiness cross-validation, the Gaussian Copula dataset elevated XGBoost 5-fold ROC-AUC from $0.5383$ to **$0.6427$** and F1-Score from $0.3987$ to **$0.5936$**.

---

## 3. Dataset Architecture & Composition

| Component | Source / Generator | Row Count | Target: Approved | Target: Rejected |
|---|---|---|---|---|
| **Empirical Seed** | Google Form Survey Responses | `500` (rows 0–499) | `232` (46.40%) | `268` (53.60%) |
| **Synthetic Augmentation** | Gaussian Copula Synthesizer | `2,500` (rows 500–2,999) | `1,164` (46.56%) | `1,336` (53.44%) |
| **Total Benchmark (v2)** | `loan_evaluation_dataset_3000_v2.csv` | **`3,000`** | **`1,396` (46.53%)** | **`1,604` (53.47%)** |

- **Integrity Validation:** 3,000 rows, 17 columns, 0 null/missing values, 0 duplicate rows.

---

## 4. Train / Test Partitioning Protocol

- **Split Ratio:** 80.0% Training ($N=2,400$) / 20.0% Holdout Testing ($N=600$).
- **Stratification:** Stratified by encoded target variable $y$ (`stratify=y`, `random_state=42`).
- **Data Partitions:**
  - **Training Matrix (`X_train`, `y_train`):** $2,400$ samples — Approved: `1,117` (46.54%), Rejected: `1,283` (53.46%).
  - **Testing Matrix (`X_test`, `y_test`):** $600$ samples — Approved: `279` (46.50%), Rejected: `321` (53.50%).

> [!CAUTION]
> **Strict Data Leakage Prevention:**
> The `ColumnTransformer` preprocessor is fitted **exclusively on `X_train`**. The test matrix `X_test` is transformed strictly using the pre-fitted transformer. The holdout test partition remains completely isolated from training, preprocessing estimation, feature selection, and hyperparameter tuning.

---

## 5. Feature Encoding Strategy

The 16 raw predictor features are processed into **55 numerical features** via a dual-encoder `ColumnTransformer`:

### 5.1 Ordinal Features (6 Features $\to$ 6 Transformed Features)
Transformed with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` using explicit financial/demographic tiers:
1. `Age Group` (6 tiers: `Below 25` $\to$ `65 or above`)
2. `Number of Dependents` (6 tiers: `0` $\to$ `5 or more`)
3. `Employment Duration` (6 tiers: `Not employed/business` $\to$ `More than 10 years`)
4. `Monthly Income` (8 tiers: `Below Rs.50,000` $\to$ `Rs.1,000,000 or above`)
5. `Monthly Loan Repayment` (7 tiers: `No existing loan` $\to$ `Rs.100,000+`)
6. `Requested Loan Amount` (9 tiers: `Below Rs.500,000` $\to$ `Rs.100,000,000 or above`)

### 5.2 Nominal Features (10 Features $\to$ 49 One-Hot Features)
Transformed with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`:
1. `Gender` (3 categories: `Female`, `Male`, `Prefer not to say`)
2. `Marital Status` (5 categories: `Divorced`, `Married`, `Prefer not to say`, `Single`, `Widowed`)
3. `Education Level` (7 categories: `Diploma`, `No formal education`, `Other`, `Postgraduate degree`, `Primary`, `Secondary`, `Undergraduate degree`)
4. `Employment Type` (8 categories: `Business owner`, `Contract employee`, `Government employee`, `Other`, `Private-sector employee`, `Retired`, `Self-employed`, `Unemployed`)
5. `Existing Loan Status` (2 categories: `No`, `Yes`)
6. `Loan Type` (7 categories: `Agricultural`, `Business`, `Education`, `Housing`, `Other`, `Personal`, `Vehicle`)
7. `Loan Purpose` (9 categories: `Business investment`, `Debt consolidation`, `Education`, `Emergency / Medical`, `Event / Wedding / Travel`, `House purchase/construction`, `Other`, `Personal / Consumption`, `Vehicle purchase`)
8. `Collateral Availability` (3 categories: `No`, `Not applicable / Not required`, `Yes`)
9. `Guarantor Availability` (2 categories: `No`, `Yes`)
10. `Required Documents Submitted` (3 categories: `No, some missing`, `Not sure`, `Yes, all`)

---

## 6. Processed Artifacts & Outputs

| Output File Path | Matrix Shape | Description |
|---|---|---|
| [data/processed/X_train.csv](file:///d:/loan-agentic-ai/data/processed/X_train.csv) | `2,400 × 55` | Preprocessed training features |
| [data/processed/X_test.csv](file:///d:/loan-agentic-ai/data/processed/X_test.csv) | `600 × 55` | Preprocessed test features |
| [data/processed/y_train.csv](file:///d:/loan-agentic-ai/data/processed/y_train.csv) | `2,400 × 1` | Encoded training target labels (`Approved`=1, `Rejected`=0) |
| [data/processed/y_test.csv](file:///d:/loan-agentic-ai/data/processed/y_test.csv) | `600 × 1` | Encoded test target labels (`Approved`=1, `Rejected`=0) |
| `data/processed/preprocessor.joblib` | — | Serialized fitted `ColumnTransformer` artifact |
| [data/processed/feature_names.json](file:///d:/loan-agentic-ai/data/processed/feature_names.json) | 55 entries | Canonical transformed feature name list |
| [evaluation/final_dataset_v2_summary.json](file:///d:/loan-agentic-ai/evaluation/final_dataset_v2_summary.json) | — | Structured dataset v2 metadata summary |

---

## 7. Next Research Steps
With the final dataset v2 preprocessed, the research pipeline is positioned for:
1. **Retraining the Random Forest Baseline Model** on the improved dataset v2 ($2,400 \times 55$).
2. **Retraining the Proposed XGBoost Classification Model** on dataset v2.
3. **Comparative Evaluation** and formal benchmarking on the holdout test set ($600 \times 55$).
4. **Post-Hoc Explainability (SHAP & LIME)** and **Agentic AI Policy Grounding (RAG)**.
