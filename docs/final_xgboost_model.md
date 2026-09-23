# Final XGBoost Proposed Model Evaluation Report (Dataset v2)

## 1. Executive Summary & Research Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 10 — Final XGBoost Proposed Model Implementation & Evaluation (Dataset v2)**
- **Objective:** Establish the final proposed Extreme Gradient Boosting (**XGBoost**) classification model trained on the preprocessed 3,000-record dataset v2 (`loan_evaluation_dataset_3000_v2.csv`). XGBoost serves as the core predictive probability engine powering downstream Explainable AI (SHAP/LIME), policy verification (RAG), and Multi-Agent decision workflows.

> [!IMPORTANT]
> **Academic Integrity & Decision Support Scope:**
> 1. The machine learning model is a predictive scoring component of a human-in-the-loop Decision Support System, not an autonomous loan approval authority.
> 2. The dataset combines **500 empirical survey responses** with **2,500 Gaussian Copula synthetic observations**; it does not represent proprietary commercial bank customer accounts.
> 3. Predictions and feature importances indicate empirical statistical correlations, not causal relationships or official institutional underwriting policies.

---

## 2. Why XGBoost is the Proposed Model
1. **Sequential Second-Order Gradient Boosting:** XGBoost sequentially minimizes a regularized logistic loss function using first and second-order Taylor approximations (gradients and hessians), allowing the model to effectively learn non-linear boundaries across multi-dimensional financial attributes.
2. **Calibrated Continuous Posterior Probabilities:** XGBoost outputs continuous probabilities $\hat{P}(\text{Approved}=1 \mid \mathbf{x})$, enabling threshold adjustments aligned with institutional risk policies.
3. **Exact Polynomial-Time TreeSHAP Compatibility:** XGBoost's tree structure allows mathematically exact, polynomial-time Shapley value computation ($O(TLD^2)$), ensuring fast, transparent local and global attribution for regulatory explainability.

---

## 3. Dataset Architecture & Class Weight Decision

- **Primary Research Benchmark:** `data/synthetic/loan_evaluation_dataset_3000_v2.csv` (3,000 records, 17 columns)
- **Preprocessed Inputs:**
  - Training Features: `data/processed/X_train.csv` ($2,400 \times 55$)
  - Testing Features: `data/processed/X_test.csv` ($600 \times 55$)
  - Training Target: `data/processed/y_train.csv` ($2,400 \times 1$)
  - Testing Target: `data/processed/y_test.csv` ($600 \times 1$)

### Class Weight Decision:
In `y_train` ($N=2,400$), class distribution is slightly asymmetric:
- Negative Class (`Rejected` = 0): $1,283$ ($53.46\%$)
- Positive Class (`Approved` = 1): $1,117$ ($46.54\%$)
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative training}}}{N_{\text{positive training}}} = \frac{1,283}{1,117} \approx \mathbf{1.1486}$$
Applying `scale_pos_weight = 1.1486` appropriately calibrates gradient step penalties for positive instances without overcompensating on balanced data.

---

## 4. Model Configuration & Hyperparameters

Deterministic, reproducible configuration:

```python
XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.1486,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `"binary:logistic"` | Logistic regression loss function for binary classification. |
| `n_estimators` | `200` | Number of sequential boosting iterations. |
| `max_depth` | `4` | Moderate tree depth balancing feature interaction capture and overfitting control. |
| `learning_rate` ($\eta$) | `0.05` | Conservative shrinkage rate to prevent premature convergence. |
| `subsample` | `0.8` | Stochastic row subsampling per tree for generalization. |
| `colsample_bytree` | `0.8` | Feature column subsampling per boosting round. |
| `scale_pos_weight` | `1.1486` | Training set negative-to-positive class ratio ($1283 / 1117$). |
| `random_state` | `42` | Ensures deterministic experimental reproducibility. |
| `eval_metric` | `"logloss"` | Logarithmic loss metric for probability optimization. |

---

## 5. Holdout Test Set Experimental Results ($N=600$)

### 5.1 Performance Metrics

| Evaluation Metric | Score | Percentage |
|---|---|---|
| **Accuracy** | `0.6200` | **62.00%** |
| **Precision (Approved / Positive Class)** | `0.5825` | **58.25%** |
| **Recall (Approved / Positive Class)** | `0.6452` | **64.52%** |
| **F1-Score (Approved / Positive Class)** | `0.6122` | **61.22%** |
| **ROC-AUC Score** | `0.6468` | **64.68%** |
| **Average Precision (PR-AUC)** | `0.5818` | **58.18%** |
| **No-Skill Positive Prevalence** | `0.4650` | 46.50% |

---

### 5.2 Confusion Matrix Breakdown

$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=192 & \text{FP}=129 \\ \text{FN}=99 & \text{TP}=180 \end{pmatrix}$$

| | Predicted: Rejected (0) | Predicted: Approved (1) | Total Actual |
|---|---|---|---|
| **Actual: Rejected (0)** | **192** (`TN`) | **129** (`FP`) | 321 |
| **Actual: Approved (1)** | **99** (`FN`) | **180** (`TP`) | 279 |
| **Total Predicted** | 291 | 309 | 600 |

- **True Positives (TP = 180):** Correct approvals (+29 compared to Random Forest baseline).
- **True Negatives (TN = 192):** Correct rejections of high-risk applicants.
- **False Positives (FP = 129):** High-risk applicants mistakenly predicted as approved (Credit risk exposure).
- **False Negatives (FN = 99):** Creditworthy applicants incorrectly predicted as rejected (-29 compared to Random Forest).

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

Rejected (0)       0.66      0.60      0.63       321
Approved (1)       0.58      0.65      0.61       279

    accuracy                           0.62       600
   macro avg       0.62      0.62      0.62       600
weighted avg       0.62      0.62      0.62       600
```

---

## 6. Cross-Validation Stability Analysis (5-Fold Stratified CV on $X_{\text{train}}$, $N=2,400$)

Cross-validation was conducted strictly on the training partition:

| Metric | Mean $\pm$ Std | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|
| **Accuracy** | **$0.5992 \pm 0.0220$** | 0.6271 | 0.6000 | 0.5979 | 0.5604 | 0.6104 |
| **Precision** | **$0.5643 \pm 0.0244$** | 0.5957 | 0.5661 | 0.5568 | 0.5231 | 0.5796 |
| **Recall** | **$0.6186 \pm 0.0236$** | 0.6250 | 0.6116 | 0.6592 | 0.6099 | 0.5874 |
| **F1-Score** | **$0.5897 \pm 0.0165$** | 0.6100 | 0.5880 | 0.6037 | 0.5631 | 0.5835 |
| **ROC-AUC** | **$0.6460 \pm 0.0261$** | 0.6828 | 0.6496 | 0.6436 | 0.6015 | 0.6526 |

---

## 7. Diagnostic Visualizations

### 7.1 Receiver Operating Characteristic (ROC) Curve
![Final XGBoost ROC Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_xgboost_roc_curve.png)

- **Holdout Test ROC-AUC:** `0.6468` (vs. Random Chance `0.5000`).

### 7.2 Precision-Recall (PR) Curve
![Final XGBoost Precision-Recall Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_xgboost_precision_recall_curve.png)

- **Holdout Test Average Precision (AP):** `0.5818` (vs. No-Skill Prevalence `0.4650`).

---

## 8. Native Feature Importance (Gain Metric)

| Rank | Feature Name | Gain Importance | Normalized Share (%) |
|---|---|---|---|
| 1 | `Required Documents Submitted_Yes, all` | `0.0865` | **8.65%** |
| 2 | `Existing Loan Status_No` | `0.0297` | **2.97%** |
| 3 | `Loan Type_Education` | `0.0276` | **2.76%** |
| 4 | `Required Documents Submitted_No, some missing` | `0.0246` | **2.46%** |
| 5 | `Guarantor Availability_Yes` | `0.0216` | **2.16%** |
| 6 | `Gender_Male` | `0.0202` | **2.02%** |
| 7 | `Collateral Availability_No` | `0.0201` | **2.01%** |
| 8 | `Loan Purpose_Debt consolidation` | `0.0199` | **1.99%** |
| 9 | `Employment Type_Self-employed` | `0.0198` | **1.98%** |
| 10 | `Requested Loan Amount` | `0.0188` | **1.88%** |

---

## 9. Comparative Summary: Random Forest Baseline vs. XGBoost Proposed

| Metric | Random Forest (Final Baseline) | XGBoost (Final Proposed) | Absolute Delta ($\Delta$) | Superior Model |
|---|---|---|---|---|
| **Accuracy** | `0.5867` | **`0.6200`** | **`+0.0333` (+3.33%)** | **XGBoost** |
| **Precision** | `0.5572` | **`0.5825`** | **`+0.0253` (+2.53%)** | **XGBoost** |
| **Recall (Approved)** | `0.5412` | **`0.6452`** | **`+0.1040` (+10.40%)** | **XGBoost** |
| **F1-Score (Approved)** | `0.5491` | **`0.6122`** | **`+0.0631` (+6.31%)** | **XGBoost** |
| **ROC-AUC** | `0.6290` | **`0.6468`** | **`+0.0178` (+1.78%)** | **XGBoost** |
| **Average Precision** | `0.5572` | **`0.5818`** | **`+0.0246` (+2.46%)** | **XGBoost** |

---

## 10. Model Selection Decision
**Option A Selected:** XGBoost is confirmed as the superior proposed prediction model, outperforming the Random Forest baseline across all holdout evaluation metrics while offering direct TreeSHAP integration for downstream Explainable AI.

---

## 11. Artifacts Generated
1. **Model Binary:** `models/xgboost/final_xgboost.joblib`
2. **Metrics Summary:** `evaluation/final_xgboost_results.json`
3. **Confusion Matrix:** `evaluation/final_xgboost_confusion_matrix.csv`
4. **Classification Report:** `evaluation/final_xgboost_classification_report.json`
5. **Predictions:** `evaluation/final_xgboost_predictions.csv`
6. **Cross-Validation Table:** `evaluation/final_xgboost_cross_validation.csv`
7. **Feature Importance:** `evaluation/final_xgboost_feature_importance.csv`
8. **Visualizations:** `final_xgboost_roc_curve.png`, `final_xgboost_precision_recall_curve.png`
9. **Model Comparison:** `evaluation/final_model_comparison.csv`, `docs/final_model_comparison.md`
