# Data Preprocessing Configuration & Documentation (Revised 3,000-Record Dataset)

## 1. Research & Experiment Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 3 — Data Preprocessing & Reproducible Feature Engineering**
- **Objective:** Transform raw categorical synthetic loan records (500 Google Form responses + 2,500 SDV CTGAN synthetic records) into numeric feature representations suitable for machine learning classification models (Random Forest baseline and XGBoost), while ensuring strict data leakage prevention and mathematical reproducibility.

---

## 2. Dataset Information & Source
- **Input File:** `data/synthetic/loan_evaluation_dataset_3000.csv`
- **Total Records:** 3,000
- **Predictor Features:** 16
- **Target Feature:** `Loan Application Result`
- **Data Quality:** 0 missing values, 0 duplicate records, 0 personally identifiable information (PII).

### Legacy Reference
- **Legacy File:** `data/synthetic/loan_evaluation_synthetic_2000_up_to_100m.csv`
- **Status:** Marked as **LEGACY / REFERENCE ONLY** (not used in active pipeline).

---

## 3. Target Variable Encoding
- **Target Column:** `Loan Application Result`
- **Mapping Strategy:**
  - `Approved` $\rightarrow$ **`1`** (Positive class)
  - `Rejected` $\rightarrow$ **`0`** (Negative class)
- **Target Distribution:**
  - Total: 1,045 Approved (34.83%) / 1,955 Rejected (65.17%)
  - Training Split ($N=2,400$): 836 Approved (34.83%) / 1,564 Rejected (65.17%)
  - Test Split ($N=600$): 209 Approved (34.83%) / 391 Rejected (65.17%)

---

## 4. Train / Test Split Protocol
- **Partition Ratio:** 80% Training (`2,400` records), 20% Testing (`600` records)
- **Split Configuration:**
  - `test_size`: `0.20`
  - `random_state`: `42`
  - `stratify`: `y` (preserves identical 34.83% : 65.17% class balance in both partitions)
- **Data Leakage Prevention:**
  - The train/test split is strictly executed **before** fitting any encoders.
  - The preprocessing transformer (`ColumnTransformer`) is fitted exclusively on `X_train`.
  - `X_test` is transformed using the fitted training preprocessor without refitting or recomputing any statistics.

---

## 5. Feature Encoding Strategy

### 5.1 Ordinal Features (Explicit Hierarchical Mapping)
Features possessing a natural, monotonic, or magnitude-based relationship are encoded using `scikit-learn.preprocessing.OrdinalEncoder` with explicitly declared orderings.

| # | Feature Name | Categories (Ascending Order: Index 0 to N-1) |
|---|---|---|
| 1 | **Age Group** | `['Below 25', '25–34', '35–44', '45–54', '55–64', '65 or above']` |
| 2 | **Number of Dependents** | `['0', '1', '2', '3', '4', '5 or more']` |
| 3 | **Employment Duration** | `['Not employed/business', 'Less than 1 year', '1–2 years', '3–5 years', '6–10 years', 'More than 10 years']` |
| 4 | **Monthly Income** | `['Below Rs.50,000', 'Rs.50,000–99,999', 'Rs.100,000–149,999', 'Rs.150,000–249,999', 'Rs.250,000–399,999', 'Rs.400,000–599,999', 'Rs.600,000–999,999', 'Rs.1,000,000 or above']` |
| 5 | **Monthly Loan Repayment** | `['No existing loan', 'Below Rs.10,000', 'Rs.10,000–24,999', 'Rs.25,000–49,999', 'Rs.50,000–74,999', 'Rs.75,000–99,999', 'Rs.100,000+']` |
| 6 | **Requested Loan Amount** | `['Below Rs.500,000', 'Rs.500,000–999,999', 'Rs.1,000,000–4,999,999', 'Rs.5,000,000–9,999,999', 'Rs.10,000,000–24,999,999', 'Rs.25,000,000–49,999,999', 'Rs.50,000,000–74,999,999', 'Rs.75,000,000–99,999,999', 'Rs.100,000,000 or above']` |

*Ordinal Out-of-Vocabulary Handling:* `handle_unknown='use_encoded_value'`, `unknown_value=-1`.

---

### 5.2 Nominal Features (One-Hot Encoding)
Nominal features without intrinsic order are encoded using `scikit-learn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

| # | Feature Name | Categories | Output Dimensions |
|---|---|---|---|
| 1 | **Gender** | `Female`, `Male`, `Prefer not to say` | 3 |
| 2 | **Marital Status** | `Divorced`, `Married`, `Prefer not to say`, `Single`, `Widowed` | 5 |
| 3 | **Education Level** | `Diploma`, `No formal education`, `Other`, `Postgraduate degree`, `Primary`, `Secondary`, `Undergraduate degree` | 7 |
| 4 | **Employment Type** | `Business owner`, `Contract employee`, `Government employee`, `Other`, `Private-sector employee`, `Retired`, `Self-employed`, `Unemployed` | 8 |
| 5 | **Existing Loan Status** | `No`, `Yes` | 2 |
| 6 | **Loan Type** | `Agricultural`, `Business`, `Education`, `Housing`, `Other`, `Personal`, `Vehicle` | 7 |
| 7 | **Loan Purpose** | `Agriculture`, `Business investment`, `Debt consolidation`, `Education`, `House purchase/construction`, `Medical expenses`, `Other`, `Personal expenses`, `Vehicle purchase` | 9 |
| 8 | **Collateral Availability** | `No`, `Not applicable / Not required`, `Yes` | 3 |
| 9 | **Guarantor Availability** | `No`, `Yes` | 2 |
| 10 | **Required Documents Submitted** | `No, some missing`, `Not sure`, `Yes, all` | 3 |

*Total One-Hot Encoded Columns:* **49 columns**  
*Total Transformed Features:* **55 columns** (6 Ordinal + 49 One-Hot)

---

### 5.3 Special Value Handling Policy
- **`Prefer not to say`** and **`Other`** values are treated as legitimate, valid domain categorical responses. They are explicitly preserved as distinct binary indicators in the one-hot encoding representation and are never treated as missing values or dropped.

---

## 6. Output Artifacts Directory: `data/processed/`

| Filename | Type / Format | Shape | Description |
|---|---|---|---|
| **`X_train.csv`** | CSV (UTF-8) | `(2400, 55)` | Preprocessed training features |
| **`X_test.csv`** | CSV (UTF-8) | `(600, 55)` | Preprocessed test features |
| **`y_train.csv`** | CSV (UTF-8) | `(2400, 1)` | Encoded training target labels (`1` / `0`) |
| **`y_test.csv`** | CSV (UTF-8) | `(600, 1)` | Encoded test target labels (`1` / `0`) |
| **`preprocessor.joblib`** | Joblib Serialized Pipeline | — | Fitted `ColumnTransformer` object for inference |
| **`feature_names.json`** | JSON Array | 55 items | Ordered list of final feature column names |

---

## 7. Model Compatibility & Downstream Readiness
The fitted `ColumnTransformer` is fully serializable and reusable across downstream modules:
1. **Random Forest Classifier (Baseline Model)**
2. **XGBoost Classifier (Proposed Model)**
3. **SHAP (TreeExplainer)**
4. **LIME (LimeTabularExplainer)**
5. **Agentic AI Decision Support Pipeline** (runtime dynamic inference)
