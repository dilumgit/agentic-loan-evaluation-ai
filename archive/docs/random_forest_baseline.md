# Random Forest Baseline Model Evaluation Report (Revised 3,000-Record Dataset)

## 1. Research Context & Purpose of the Baseline
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 4 — Baseline Model Implementation & Academic Evaluation (Revised Dataset)**
- **Objective:** Establish an empirical performance baseline using a standard tree-ensemble classifier (Random Forest) on the newly established primary **3,000-record research dataset** (500 empirical Google Form seed responses + 2,500 SDV CTGAN synthetic records). This baseline serves as the benchmark against which the proposed gradient-boosted model (XGBoost) and subsequent Explainable AI (SHAP, LIME) and Agentic AI modules will be compared.

> [!IMPORTANT]
> **Dataset Lineage & Legacy Disclaimer:**
> The previous 2,000-record synthetic dataset and its associated classification results are strictly designated as **LEGACY / HISTORICAL BENCHMARKS**. The metrics documented herein represent the active baseline on the current 3,000-record research benchmark (`data/synthetic/loan_evaluation_dataset_3000.csv`).

---

## 2. Dataset & Partitioning Protocol
- **Primary Research Benchmark:** `data/synthetic/loan_evaluation_dataset_3000.csv` (3,000 rows, 17 columns)
- **Preprocessed Inputs:**
  - Training Features: `data/processed/X_train.csv` ($2,400 \times 55$)
  - Testing Features: `data/processed/X_test.csv` ($600 \times 55$)
  - Training Target: `data/processed/y_train.csv` ($2,400 \times 1$)
  - Testing Target: `data/processed/y_test.csv` ($600 \times 1$)
- **Data Partitions (Stratified 80/20 Split, `random_state=42`):**
  - **Train Set ($N=2,400$, 80.0%):** 836 Approved (34.83%), 1,564 Rejected (65.17%)
  - **Test Set ($N=600$, 20.0%):** 209 Approved (34.83%), 391 Rejected (65.17%)
- **Feature Space:** 55 preprocessed numerical features (6 ordinal features + 49 one-hot indicator columns).

---

## 3. Model Configuration & Hyperparameters

The Random Forest baseline model was instantiated with reproducible hyperparameters without hyperparameter tuning:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

| Hyperparameter | Value | Academic Justification |
|---|---|---|
| `n_estimators` | `200` | Standard ensemble size providing variance reduction and stable bagging aggregation. |
| `random_state` | `42` | Ensures deterministic pseudo-random tree generation and mathematical reproducibility. |
| `class_weight` | `"balanced"` | Automatically adjusts weights inversely proportional to class frequencies ($65.17\%$ Rejected vs. $34.83\%$ Approved). |
| `n_jobs` | `-1` | Parallelized multi-core training for computational efficiency. |

---

## 4. Evaluation Metrics & Experimental Results

Evaluated on the un-seen holdout test dataset ($N = 600$):

### 4.1 Summary Metrics Table

| Metric | Score | Percentage |
|---|---|---|
| **Accuracy** | `0.6033` | **60.33%** |
| **Precision (Approved / Positive Class)** | `0.3986` | **39.86%** |
| **Recall (Approved / Positive Class)** | `0.2727` | **27.27%** |
| **F1-Score (Approved / Positive Class)** | `0.3239` | **32.39%** |
| **ROC-AUC** | `0.5518` | **55.18%** |

---

### 4.2 Confusion Matrix Analysis

$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=305 & \text{FP}=86 \\ \text{FN}=152 & \text{TP}=57 \end{pmatrix}$$

| | Predicted: Rejected (0) | Predicted: Approved (1) | Total Actual |
|---|---|---|---|
| **Actual: Rejected (0)** | **305** (TN) | **86** (FP) | 391 |
| **Actual: Approved (1)** | **152** (FN) | **57** (TP) | 209 |
| **Total Predicted** | 457 | 143 | 600 |

#### Quantitative Risk Interpretation:
- **True Positives (TP = 57):** Correctly approved creditworthy applicants.
- **True Negatives (TN = 305):** Correctly identified and rejected risky applications.
- **False Positives (FP = 86):** High-risk applicants incorrectly predicted as approved (Credit default risk).
- **False Negatives (FN = 152):** Creditworthy applicants incorrectly predicted as rejected (Customer friction / opportunity loss).

---

### 4.3 Detailed Classification Report

```
              precision    recall  f1-score   support

Rejected (0)       0.67      0.78      0.72       391
Approved (1)       0.40      0.27      0.32       209

    accuracy                           0.60       600
   macro avg       0.53      0.53      0.52       600
weighted avg       0.57      0.60      0.58       600
```

---

## 5. Comparison with Legacy Baseline (Academic Reference Only)

| Dimension | Legacy 2,000 Dataset Baseline | Current 3,000 Dataset Baseline |
|---|---|---|
| **Dataset Source** | Pure synthetic generated (2,000 records) | Empirical 500 Google Form Seed + 2,500 SDV CTGAN Synthetic (3,000 records) |
| **Class Balance** | Perfectly balanced (50.0% / 50.0%) | Empirical distribution (34.83% Approved / 65.17% Rejected) |
| **Training / Test Size** | 1,600 / 400 | **2,400 / 600** |
| **Baseline Accuracy** | 78.50% *(Legacy)* | **60.33%** |
| **Baseline F1-Score** | 78.71% *(Legacy)* | **32.39%** |
| **Baseline ROC-AUC** | 0.8676 *(Legacy)* | **0.5518** |

*Note: The shift reflects the genuine complexity and realistic distribution captured from real Google Form respondents and CTGAN synthesis, highlighting the strong academic justification for advanced gradient boosting (XGBoost), feature tuning, explainability (SHAP, LIME), and agentic policy grounding in subsequent phases.*

---

## 6. Artifacts Generated

1. **Model Binary:** `models/baseline/random_forest.joblib`
2. **Metrics Summary:** `evaluation/random_forest_results.json`
3. **Confusion Matrix:** `evaluation/random_forest_confusion_matrix.csv`
4. **Predictions:** `evaluation/random_forest_predictions.csv`
5. **Standalone Validator Script:** `evaluation/evaluate_baseline.py`

---

## 7. Limitations of the Baseline Model

1. **Weak Discrimination on Imbalanced Minority Class:** Standard un-tuned Random Forest struggles to capture the complex boundary of the minority `Approved` class ($34.83\%$), resulting in low recall ($27.27\%$).
2. **Bagging Variance vs. Gradient Boosting:** Independent tree aggregation cannot iteratively optimize hard-to-classify edge cases as effectively as XGBoost.
3. **Absence of Regulatory Policy Context:** The model makes purely statistical predictions and lacks compliance mechanisms to enforce banking lending criteria.
4. **Lack of Transparent Explanations:** Bagged ensembles do not yield local attribution explanations for loan officers without integrated XAI frameworks.
