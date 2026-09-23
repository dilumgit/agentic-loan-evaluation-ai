# CatBoost Benchmark Model Evaluation & 3-Model Comparative Analysis

## 1. Executive Summary & Research Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 11 — CatBoost Benchmark Model (Controlled Academic Experiment)**
- **Objective:** Evaluate CatBoost (`catboost.CatBoostClassifier`) as a third candidate model on the preprocessed 3,000-record dataset v2 (`loan_evaluation_dataset_3000_v2.csv`). Rigorously benchmark CatBoost against the established **Random Forest Baseline** and **XGBoost Proposed Model** under identical experimental conditions.
- **Core Research Question:** *“Does a categorical boosting approach using CatBoost provide improved predictive performance for the bank loan evaluation dataset compared with Random Forest and XGBoost?”*

> [!IMPORTANT]
> **Academic Integrity & Decision Support Scope:**
> 1. All three classifiers (Random Forest, XGBoost, CatBoost) function as predictive scoring algorithms within a human-in-the-loop Decision Support System; none act as autonomous loan approval authorities.
> 2. The dataset comprises **500 empirical survey questionnaire responses** combined with **2,500 Gaussian Copula synthetic records**; it does not represent actual commercial bank customer accounts.
> 3. Feature importance scores indicate empirical correlations within this dataset, not causal relationships or official underwriting policies.

---

## 2. Why CatBoost Was Evaluated
1. **Symmetric Oblivious Trees:** CatBoost builds balanced, symmetric decision trees (oblivious trees) that use the same split criterion across all nodes at the same tree depth, reducing variance and offering strong execution speed.
2. **Ordered Boosting Mechanism:** CatBoost implements ordered boosting to combat target leakage (prediction shift) during gradient calculation on tabular datasets.
3. **Benchmarking Objective:** To verify whether CatBoost's inductive biases yield performance gains over standard gradient boosting (XGBoost) and bagging (Random Forest) on this preprocessed financial feature space.

---

## 3. Dataset & Experimental Controls
- **Dataset Source:** `data/synthetic/loan_evaluation_dataset_3000_v2.csv` (3,000 rows, 17 columns)
- **Train/Test Partitions (80/20 Stratified Split, `random_state=42`):**
  - **Training Matrix:** `data/processed/X_train.csv` ($2,400 \times 55$), `y_train.csv` ($2,400 \times 1$)
  - **Holdout Test Matrix:** `data/processed/X_test.csv` ($600 \times 55$), `y_test.csv` ($600 \times 1$)
- **Data Leakage Isolation:** Preprocessor was fitted strictly on $X_{\text{train}}$. The holdout test set ($N=600$) remained untouched until final evaluation.

---

## 4. CatBoost Model Configuration & Class Imbalance Handling

### 4.1 Hyperparameter Specifications

```python
CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    scale_pos_weight=1.1486,
    random_seed=42,
    verbose=False,
    thread_count=-1
)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `iterations` | `300` | Ample boosting rounds to achieve loss convergence. |
| `depth` | `6` | Standard tree depth for symmetric oblivious trees. |
| `learning_rate` ($\eta$) | `0.05` | Conservative shrinkage rate to prevent overfitting. |
| `loss_function` | `"Logloss"` | Binary cross-entropy logistic loss. |
| `eval_metric` | `"AUC"` | Area Under the ROC Curve for binary optimization. |
| `scale_pos_weight` | `1.1486` | Ratio of negative ($1,283$) to positive ($1,117$) training samples. |
| `random_seed` | `42` | Ensures deterministic experimental reproducibility. |
| `thread_count` | `-1` | Parallelized multi-core execution. |

### 4.2 Class Imbalance Handling
In `y_train` ($N=2,400$), negative samples (`Rejected` = 0) equal $1,283$ ($53.46\%$) and positive samples (`Approved` = 1) equal $1,117$ ($46.54\%$). The positive class penalty was set to:
$$\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{1,283}{1,117} \approx \mathbf{1.1486}$$
calculated strictly on $X_{\text{train}}$ without exposure to $X_{\text{test}}$.

---

## 5. Holdout Test Set Performance ($N=600$)

| Evaluation Metric | Random Forest (Baseline) | XGBoost (Proposed Model) | CatBoost (Benchmark) | Best Performing Model |
|---|---|---|---|---|
| **Accuracy** | `0.5867` (58.67%) | **`0.6200` (62.00%)** | `0.6033` (60.33%) | **XGBoost** |
| **Precision (Approved)** | `0.5572` (55.72%) | **`0.5825` (58.25%)** | `0.5686` (56.86%) | **XGBoost** |
| **Recall (Approved)** | `0.5412` (54.12%) | **`0.6452` (64.52%)** | `0.6093` (60.93%) | **XGBoost** |
| **F1-Score (Approved)** | `0.5491` (54.91%) | **`0.6122` (61.22%)** | `0.5882` (58.82%) | **XGBoost** |
| **ROC-AUC Score** | `0.6290` (62.90%) | **`0.6468` (64.68%)** | `0.6236` (62.36%) | **XGBoost** |
| **Average Precision (AP)** | `0.5572` (55.72%) | **`0.5818` (58.18%)** | `0.5570` (55.70%) | **XGBoost** |

---

## 6. Confusion Matrix & Error Breakdown ($N=600$ Test Set)

$$\text{Random Forest} = \begin{pmatrix} \text{TN}=201 & \text{FP}=120 \\ \text{FN}=128 & \text{TP}=151 \end{pmatrix}, \quad \text{XGBoost} = \begin{pmatrix} \text{TN}=192 & \text{FP}=129 \\ \text{FN}=99 & \text{TP}=180 \end{pmatrix}, \quad \text{CatBoost} = \begin{pmatrix} \text{TN}=192 & \text{FP}=129 \\ \text{FN}=109 & \text{TP}=170 \end{pmatrix}$$

| Metric / Category | Random Forest (Baseline) | XGBoost (Proposed) | CatBoost (Benchmark) | Banking Interpretation |
|---|---|---|---|---|
| **True Positives (`TP`)** | 151 | **180** | 170 | XGBoost captures the highest number of creditworthy applicants (+10 vs. CatBoost, +29 vs. RF). |
| **True Negatives (`TN`)** | **201** | 192 | 192 | Random Forest rejects the highest number of high-risk applicants. |
| **False Positives (`FP`)** | **120** | 129 | 129 | XGBoost and CatBoost yield identical false positive counts (129). |
| **False Negatives (`FN`)** | 128 | **99** | 109 | **XGBoost minimizes wrongful rejections of creditworthy applicants (99 vs. 109 in CatBoost, 128 in RF).** |

#### Detailed CatBoost Classification Report:
```
              precision    recall  f1-score   support

Rejected (0)       0.64      0.60      0.62       321
Approved (1)       0.57      0.61      0.59       279

    accuracy                           0.60       600
   macro avg       0.60      0.60      0.60       600
weighted avg       0.60      0.60      0.60       600
```

---

## 7. 5-Fold Stratified Cross-Validation on $X_{\text{train}}$ ($N=2,400$)

| Evaluation Metric | Random Forest (Mean $\pm$ Std) | XGBoost (Mean $\pm$ Std) | CatBoost (Mean $\pm$ Std) | Best CV Stability |
|---|---|---|---|---|
| **Accuracy** | **$0.6088 \pm 0.0220$** | $0.5992 \pm 0.0220$ | $0.5858 \pm 0.0279$ | Random Forest |
| **Precision** | **$0.5799 \pm 0.0273$** | $0.5643 \pm 0.0244$ | $0.5528 \pm 0.0305$ | Random Forest |
| **Recall** | $0.5864 \pm 0.0310$ | **$0.6186 \pm 0.0236$** | $0.5935 \pm 0.0140$ | **XGBoost** |
| **F1-Score** | $0.5824 \pm 0.0202$ | **$0.5897 \pm 0.0165$** | $0.5719 \pm 0.0182$ | **XGBoost** |
| **ROC-AUC** | **$0.6460 \pm 0.0238$** | **$0.6460 \pm 0.0261$** | $0.6348 \pm 0.0219$ | **XGBoost & RF tied** |

---

## 8. Diagnostic Visualizations

### 8.1 Receiver Operating Characteristic (ROC) Curve
![CatBoost ROC Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_catboost_roc_curve.png)

- **Holdout Test ROC-AUC:** `0.6236` (vs. XGBoost `0.6468`, Random Forest `0.6290`).

### 8.2 Precision-Recall (PR) Curve
![CatBoost Precision-Recall Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/final_catboost_precision_recall_curve.png)

- **Holdout Test Average Precision (AP):** `0.5570` (vs. XGBoost `0.5818`, Random Forest `0.5572`).

---

## 9. Native Feature Importance (CatBoost)

| Rank | Feature Name | Importance | Normalized Share (%) |
|---|---|---|---|
| 1 | `Requested Loan Amount` | `10.24` | **10.24%** |
| 2 | `Monthly Income` | `7.82` | **7.82%** |
| 3 | `Employment Duration` | `6.56` | **6.56%** |
| 4 | `Number of Dependents` | `5.74` | **5.74%** |
| 5 | `Age Group` | `5.59` | **5.59%** |
| 6 | `Monthly Loan Repayment` | `5.58` | **5.58%** |
| 7 | `Required Documents Submitted_Yes, all` | `4.27` | **4.27%** |
| 8 | `Loan Type_Personal` | `2.71` | **2.71%** |
| 9 | `Collateral Availability_No` | `2.44` | **2.44%** |
| 10 | `Existing Loan Status_Yes` | `2.01` | **2.01%** |

---

## 10. Research Findings & Final Model Selection Decision

### 10.1 Key Findings:
1. **CatBoost Improves Upon Random Forest Baseline:** CatBoost achieved higher holdout test accuracy (`60.33%` vs. `58.67%`), precision (`56.86%` vs. `55.72%`), recall (`60.93%` vs. `54.12%`), and F1-score (`58.82%` vs. `54.91%`) compared to Random Forest.
2. **XGBoost Demonstrates Consistent Superiority:** Across all 6 holdout test evaluation criteria (ROC-AUC: `64.68%`, F1: `61.22%`, Recall: `64.52%`, Accuracy: `62.00%`, Precision: `58.25%`, AP: `58.18%`), XGBoost consistently outperforms both CatBoost and Random Forest.
3. **False Negative Minimization:** XGBoost yielded the lowest false negative count ($FN=99$ vs. CatBoost $109$ and RF $128$), minimizing missed lending opportunities.

### 10.2 Model Selection Decision:
**“XGBoost remains the selected proposed model.”**

The empirical evidence from this controlled 3-model benchmark confirms that XGBoost provides the optimal trade-off of discriminative power, recall, and harmonic F1 balance for this financial loan dataset, while natively supporting fast polynomial-time TreeSHAP attributions for the upcoming Explainable AI phase.
