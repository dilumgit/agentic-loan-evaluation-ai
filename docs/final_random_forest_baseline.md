# Final Random Forest Baseline Model Evaluation Report (Dataset v2)

## 1. Research Context & Purpose of Random Forest Baseline
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 9 — Final Random Forest Baseline Model Training & Evaluation (Dataset v2)**
- **Objective:** Establish the final, rigorous machine learning baseline using an ensemble Random Forest classifier on the validated **3,000-record research dataset v2** (`loan_evaluation_dataset_3000_v2.csv`). This baseline serves as the empirical standard against which the proposed XGBoost gradient boosting classifier is evaluated.

> [!IMPORTANT]
> **Academic Integrity & Decision-Making Disclaimer:**
> 1. The Random Forest model functions solely as an empirical machine-learning baseline for academic research benchmarking.
> 2. The model does not make real-world banking decisions or enforce proprietary credit underwriting policies.
> 3. The dataset consists of **500 empirical Google Form questionnaire responses** combined with **2,500 Gaussian Copula-generated synthetic observations** and does not represent actual commercial bank customer accounts.

---

## 2. Why Random Forest is the Baseline Model
1. **Established Ensemble Bagging Standard:** Random Forest constructs a forest of decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling, providing a robust, non-linear baseline.
2. **Inherent Variance Reduction:** By averaging across 200 independent trees, Random Forest suppresses variance without high sensitivity to hyperparameter configurations.
3. **Transparent Class Weighting:** Through `class_weight="balanced"`, tree nodes adjust split criteria according to inverse class frequencies, preventing majority class bias.
4. **Natural Interpretability Baseline:** Provides Gini impurity reduction feature importances for comparative analysis with gradient-boosted gain and downstream TreeSHAP attributions.

---

## 3. Dataset & Partitioning Protocol

- **Dataset Source:** `data/synthetic/loan_evaluation_dataset_3000_v2.csv` (500 Empirical + 2,500 Gaussian Copula synthetic instances)
- **Preprocessed Inputs:**
  - Training Features: `data/processed/X_train.csv` ($2,400 \times 55$)
  - Testing Features: `data/processed/X_test.csv` ($600 \times 55$)
  - Training Target: `data/processed/y_train.csv` ($2,400 \times 1$)
  - Testing Target: `data/processed/y_test.csv` ($600 \times 1$)
- **Data Partitions (Stratified 80/20 Split, `random_state=42`):**
  - **Train Set ($N=2,400$):** `Approved`: 1,117 (46.54%), `Rejected`: 1,283 (53.46%)
  - **Test Set ($N=600$):** `Approved`: 279 (46.50%), `Rejected`: 321 (53.50%)
- **Data Leakage Isolation:** The preprocessor was fitted exclusively on $X_{\text{train}}$. The holdout test set ($N=600$) remained completely untouched until final inference.

---

## 4. Model Configuration & Hyperparameters

Deterministic, unoptimized standard baseline parameters:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | `200` | Ample ensemble size to achieve convergence in bagged variance. |
| `random_state` | `42` | Ensures exact experimental reproducibility. |
| `class_weight` | `"balanced"` | Adjusts weights inversely proportional to class frequencies. |
| `n_jobs` | `-1` | Parallelized execution across CPU cores. |

---

## 5. Holdout Test Set Experimental Results ($N=600$)

### 5.1 Performance Metrics

| Evaluation Metric | Final RF Baseline (Dataset v2) | Legacy RF Baseline (CTGAN) | Absolute Improvement ($\Delta$) |
|---|---|---|---|
| **Accuracy** | `0.5867` (**58.67%**) | `0.6033` (60.33%) | `-0.0166` (-1.66%) |
| **Precision (Approved)** | `0.5572` (**55.72%**) | `0.3986` (39.86%) | **`+0.1586` (+15.86%)** |
| **Recall (Approved)** | `0.5412` (**54.12%**) | `0.2727` (27.27%) | **`+0.2685` (+26.85%)** |
| **F1-Score (Approved)** | `0.5491` (**54.91%**) | `0.3239` (32.39%) | **`+0.2252` (+22.52%)** |
| **ROC-AUC** | `0.6290` (**62.90%**) | `0.5518` (55.18%) | **`+0.0772` (+7.72%)** |
| **Average Precision (AP)** | `0.5572` (**55.72%**) | `0.4126` (41.26%) | **`+0.1446` (+14.46%)** |
| **No-Skill Class Prevalence** | `0.4650` (46.50%) | `0.3483` (34.83%) | — |

---

### 5.2 Confusion Matrix Breakdown

$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=201 & \text{FP}=120 \\ \text{FN}=128 & \text{TP}=151 \end{pmatrix}$$

| | Predicted: Rejected (0) | Predicted: Approved (1) | Total Actual |
|---|---|---|---|
| **Actual: Rejected (0)** | **201** (`TN`) | **120** (`FP`) | 321 |
| **Actual: Approved (1)** | **128** (`FN`) | **151** (`TP`) | 279 |
| **Total Predicted** | 329 | 271 | 600 |

- **True Positives (TP = 151):** Correctly identified creditworthy applicants (+94 compared to legacy CTGAN baseline).
- **True Negatives (TN = 201):** Correctly identified high-risk applications.
- **False Positives (FP = 120):** Risky loans mistakenly approved (Credit risk exposure).
- **False Negatives (FN = 128):** Creditworthy applicants incorrectly rejected (-24 compared to legacy CTGAN baseline).

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

Rejected (0)       0.61      0.63      0.62       321
Approved (1)       0.56      0.54      0.55       279

    accuracy                           0.59       600
   macro avg       0.58      0.58      0.58       600
weighted avg       0.59      0.59      0.59       600
```

---

## 6. Cross-Validation Stability Analysis (5-Fold Stratified CV on $X_{\text{train}}$, $N=2,400$)

Cross-validation was conducted strictly on the training partition:

| Metric | Mean $\pm$ Std | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|
| **Accuracy** | **$0.6088 \pm 0.0220$** | 0.6312 | 0.6188 | 0.6125 | 0.5667 | 0.6146 |
| **Precision** | **$0.5799 \pm 0.0273$** | 0.6135 | 0.5953 | 0.5737 | 0.5319 | 0.5848 |
| **Recall** | **$0.5864 \pm 0.0310$** | 0.5670 | 0.5714 | 0.6457 | 0.5605 | 0.5874 |
| **F1-Score** | **$0.5824 \pm 0.0202$** | 0.5893 | 0.5831 | 0.6076 | 0.5459 | 0.5861 |
| **ROC-AUC** | **$0.6460 \pm 0.0238$** | 0.6827 | 0.6477 | 0.6481 | 0.6075 | 0.6442 |

**Finding:** The low standard deviations across all 5 training folds ($\sigma \approx 0.02$) demonstrate high cross-validation stability, closely aligning with holdout test metrics.

---

## 7. Diagnostic Visualizations

### 7.1 Receiver Operating Characteristic (ROC) Curve
![Final Random Forest ROC Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_random_forest_roc_curve.png)

- **Holdout Test ROC-AUC:** `0.6290` (vs. random chance `0.5000`).

### 7.2 Precision-Recall (PR) Curve
![Final Random Forest Precision-Recall Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_random_forest_precision_recall_curve.png)

- **Holdout Test Average Precision (AP):** `0.5572` (vs. no-skill positive prevalence line `0.4650`).

---

## 8. Feature Importance Diagnostic (Native Gini Impurity)

| Rank | Feature Name | Gini Importance | Normalized Share (%) |
|---|---|---|---|
| 1 | `Requested Loan Amount` | `0.0742` | **7.42%** |
| 2 | `Monthly Income` | `0.0687` | **6.87%** |
| 3 | `Employment Duration` | `0.0623` | **6.23%** |
| 4 | `Age Group` | `0.0585` | **5.85%** |
| 5 | `Number of Dependents` | `0.0583` | **5.83%** |
| 6 | `Monthly Loan Repayment` | `0.0412` | **4.12%** |
| 7 | `Required Documents Submitted_Yes, all` | `0.0229` | **2.29%** |
| 8 | `Education Level_Undergraduate degree` | `0.0213` | **2.13%** |
| 9 | `Marital Status_Married` | `0.0195` | **1.95%** |
| 10 | `Employment Type_Private-sector employee` | `0.0194` | **1.94%** |

---

## 9. Limitations & Next Steps
1. **Sub-optimal Boundary Non-Linearity:** As a bagging method, Random Forest does not adaptively optimize sequential loss gradients. This limitation will be addressed by the proposed XGBoost model in Step 10.
2. **Post-Hoc Explainability Scope:** Gini importance provides global split heuristics but cannot provide localized, mathematically sound Shapley explanations for individual loan applicants (reserved for SHAP/LIME in subsequent steps).
