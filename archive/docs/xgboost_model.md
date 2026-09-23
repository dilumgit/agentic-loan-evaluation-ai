# XGBoost Proposed Model Evaluation Report (Revised 3,000-Record Dataset)

## 1. Research Context & Purpose of XGBoost
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 5 — Proposed XGBoost Model Implementation & Comparative Evaluation (Revised Dataset)**
- **Objective:** Train, evaluate, and benchmark the proposed Extreme Gradient Boosting (**XGBoost**) classifier against the Random Forest baseline on the primary **3,000-record research dataset** (500 empirical Google Form responses + 2,500 SDV CTGAN synthetic records). XGBoost provides calibrated posterior probability estimates that feed into downstream Explainable AI (SHAP / LIME), policy retrieval (RAG), and multi-agent decision support workflows.

> [!IMPORTANT]
> **Dataset Lineage & Legacy Disclaimer:**
> The previous 2,000-record synthetic dataset and its earlier model results are designated as **LEGACY / HISTORICAL BENCHMARKS**. The metrics documented below are strictly based on the active 3,000-record dataset (`data/synthetic/loan_evaluation_dataset_3000.csv`).

---

## 2. Why XGBoost is the Proposed Prediction Model
1. **Sequential Gradient Boosting Optimization:** Unlike bagging methods (e.g., Random Forest) that construct independent trees, XGBoost sequentially minimizes regularized loss functions using second-order Taylor expansions (gradients and hessians), allowing the model to adaptively address hard-to-classify samples.
2. **Explicit Class Imbalance Handling:** XGBoost natively incorporates `scale_pos_weight`, which dynamically reweights positive class gradient steps to counter class imbalance without synthetic oversampling of the test distribution.
3. **Calibrated Continuous Probabilities:** XGBoost outputs continuous posterior probabilities $\hat{P}(\text{Approved}=1 \mid \mathbf{x})$, enabling threshold adjustments aligned with banking risk policies.
4. **Native Explainability Support:** Tree-based XGBoost models support polynomial-time TreeSHAP algorithms ($O(TLD^2)$), ensuring mathematically exact local and global Shapley attribution values for transparent regulatory compliance.

---

## 3. Dataset & Partitioning Protocol
- **Primary Research Benchmark:** `data/synthetic/loan_evaluation_dataset_3000.csv` (3,000 records, 17 columns)
- **Preprocessed Inputs:**
  - Training Features: `data/processed/X_train.csv` ($2,400 \times 55$)
  - Testing Features: `data/processed/X_test.csv` ($600 \times 55$)
  - Training Target: `data/processed/y_train.csv` ($2,400 \times 1$)
  - Testing Target: `data/processed/y_test.csv` ($600 \times 1$)
- **Data Partitions (Stratified 80/20 Split, `random_state=42`):**
  - **Train Set ($N=2,400$, 80.0%):** 836 Approved (34.83%), 1,564 Rejected (65.17%)
  - **Test Set ($N=600$, 20.0%):** 209 Approved (34.83%), 391 Rejected (65.17%)
- **Feature Space:** 55 preprocessed numerical features (6 ordinal categories + 49 one-hot dummy indicators).

---

## 4. Model Configuration & Class Imbalance Handling

### 4.1 Hyperparameter Specifications
The XGBoost classifier was trained using deterministic, unoptimized hyperparameters:

```python
XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.8708,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)
```

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `"binary:logistic"` | Logistic regression loss for binary probability estimation. |
| `n_estimators` | `200` | Number of sequential boosting rounds. |
| `max_depth` | `4` | Moderate tree depth balancing feature interaction capture and overfitting prevention. |
| `learning_rate` ($\eta$) | `0.05` | Conservative shrinkage rate to ensure stable convergence. |
| `subsample` | `0.8` | Stochastic row subsampling per tree for generalization. |
| `colsample_bytree` | `0.8` | Subsampling of feature columns per tree. |
| `scale_pos_weight` | `1.8708` | Ratio of negative to positive training samples ($1564 / 836$). |
| `random_state` | `42` | Guarantees exact experimental reproducibility. |
| `eval_metric` | `"logloss"` | Logarithmic loss metric for probability optimization. |

### 4.2 Class Imbalance Handling
The training set reflects an imbalanced target distribution with 1,564 Rejected (65.17%) and 836 Approved (34.83%). To ensure the minority `Approved` class is appropriately penalized during gradient updates:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{1,564}{836} \approx 1.8708$$
*Note: The holdout test set was neither modified nor artificially rebalanced, ensuring strict real-world evaluation integrity.*

---

## 5. Evaluation Methodology & Experimental Results

Evaluated on the un-seen holdout test dataset ($N = 600$):

### 5.1 Performance Metrics

| Metric | Score | Percentage |
|---|---|---|
| **Accuracy** | `0.5450` | **54.50%** |
| **Precision (Approved / Positive Class)** | `0.3720` | **37.20%** |
| **Recall (Approved / Positive Class)** | `0.4450` | **44.50%** |
| **F1-Score (Approved / Positive Class)** | `0.4052` | **40.52%** |
| **ROC-AUC** | `0.5422` | **54.22%** |

---

### 5.2 Confusion Matrix Breakdown

$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=234 & \text{FP}=157 \\ \text{FN}=116 & \text{TP}=93 \end{pmatrix}$$

| | Predicted: Rejected (0) | Predicted: Approved (1) | Total Actual |
|---|---|---|---|
| **Actual: Rejected (0)** | **234** (`TN`) | **157** (`FP`) | 391 |
| **Actual: Approved (1)** | **116** (`FN`) | **93** (`TP`) | 209 |
| **Total Predicted** | 350 | 250 | 600 |

- **True Positives (TP = 93):** Correctly approved creditworthy applicants (+36 compared to Random Forest).
- **True Negatives (TN = 234):** Correctly identified and rejected risky applications.
- **False Positives (FP = 157):** High-risk applications mistakenly approved (Credit risk exposure).
- **False Negatives (FN = 116):** Creditworthy applicants incorrectly rejected (-36 compared to Random Forest).

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

Rejected (0)       0.67      0.60      0.63       391
Approved (1)       0.37      0.44      0.41       209

    accuracy                           0.55       600
   macro avg       0.52      0.52      0.52       600
weighted avg       0.57      0.55      0.55       600
```

---

## 6. Comparative Analysis: Random Forest Baseline vs. XGBoost Proposed

| Metric | Random Forest (Baseline) | XGBoost (Proposed) | Absolute Difference ($\Delta$) | Direction / Impact |
|---|---|---|---|---|
| **Accuracy** | **`0.6033`** | `0.5450` | **`-0.0583` (-5.83%)** | Random Forest higher overall accuracy |
| **Precision** | **`0.3986`** | `0.3720` | **`-0.0266` (-2.66%)** | Random Forest slightly higher precision |
| **Recall (Approved)** | `0.2727` | **`0.4450`** | **`+0.1723` (+17.23%)** | **XGBoost substantially superior recall (+63% relative)** |
| **F1-Score (Approved)** | `0.3239` | **`0.4052`** | **`+0.0813` (+8.13%)** | **XGBoost superior harmonic balance (+25% relative)** |
| **ROC-AUC** | **`0.5518`** | `0.5422` | **`-0.0096` (-0.96%)** | Comparable area under curve |

### Academic Discussion of Trade-offs:
1. **Recall & Minority Class Sensitivity:** XGBoost with `scale_pos_weight=1.8708` successfully elevates sensitivity for the minority creditworthy class (`Approved`), increasing recall from **27.27% (57 TP)** in Random Forest to **44.50% (93 TP)**, reducing missed opportunities by 36 applicants.
2. **F1-Score Advantage:** By improving minority class capture, XGBoost achieves an F1-Score of **0.4052** compared to Random Forest's **0.3239** (+8.13% absolute improvement).
3. **Accuracy & Precision Trade-off:** Because Random Forest predicts the majority `Rejected` class more conservatively (305 TN vs. 234 TN), it achieves higher overall accuracy (60.33% vs. 54.50%) at the severe cost of rejecting 72.73% of creditworthy borrowers.
4. **Research Justification for Subsequent Steps:** In banking decision support systems, missed creditworthy applicants represent significant revenue loss. XGBoost provides a much better balance between precision and recall, and its continuous probability output enables downstream Agentic AI decision policies to apply custom thresholding and regulatory rule overrides.

---

## 7. Artifacts Generated

1. **Model Binary:** `models/xgboost/xgboost_model.joblib`
2. **Metrics Summary:** `evaluation/xgboost_results.json`
3. **Confusion Matrix:** `evaluation/xgboost_confusion_matrix.csv`
4. **Predictions:** `evaluation/xgboost_predictions.csv`
5. **Comparison Script:** `evaluation/compare_models.py`
6. **Comparison Summary Table:** `evaluation/model_comparison.csv`

---

## 8. Current Limitations & Next Research Phases

1. **Unoptimized Hyperparameters:** Both models were evaluated with standard un-tuned default hyperparameters. Systematic Bayesian optimization or Grid Search over `max_depth`, `learning_rate`, `subsample`, `min_child_weight`, and `gamma` could further boost discrimination.
2. **Post-Hoc Explainability (Next Stage):** TreeSHAP and LIME must be applied to explain XGBoost predictions at both the global feature level and individual loan applicant level.
3. **Policy-Aware Grounding (RAG & Agents):** Neither model incorporates statutory regulatory compliance rules (e.g., minimum income thresholds, mandatory document validation, debt-to-income limits). These will be governed by the LangChain RAG vector store and Multi-Agent decision framework in later stages.
