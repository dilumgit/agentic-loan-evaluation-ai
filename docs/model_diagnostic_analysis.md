# Model Validation & Diagnostic Analysis Report

## 1. Executive Summary & Research Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 6 — Model Validation and Diagnostic Analysis (3,000-Record Benchmark)**
- **Objective:** Conduct a comprehensive diagnostic investigation into the statistical properties of the primary 3,000-record dataset (`loan_evaluation_dataset_3000.csv` = 500 empirical Google Form responses + 2,500 SDV CTGAN synthetic records) and evaluate why both the Random Forest baseline and the proposed XGBoost classifier exhibit modest predictive performance (ROC-AUC $\approx 0.54\text{--}0.55$).

> [!IMPORTANT]
> **Academic Integrity & Data Scope Disclaimer:**
> 1. This analysis evaluates synthetic and empirical survey data collected for academic decision-support research.
> 2. **No Causal Claim:** Statistical associations (Chi-square, Cramér's V) denote empirical correlations, not causal relationships.
> 3. **No Representation of Institutional Bank Policy:** Predictions and model weights do not reflect actual proprietary underwriting rules of commercial banks.
> 4. **Distinction of Data Provenance:** We strictly distinguish between the 500 empirical Google Form survey responses and the 2,500 SDV CTGAN synthetic instances.

---

## 2. Overall Model Performance Summary (Holdout Test Set, $N=600$)

| Evaluation Metric | Random Forest (Baseline) | XGBoost (Proposed) | Absolute Delta ($\Delta$) |
|---|---|---|---|
| **Accuracy** | `0.6033` (60.33%) | `0.5450` (54.50%) | `-0.0583` (-5.83%) |
| **Precision (Approved / Positive Class)** | `0.3986` (39.86%) | `0.3720` (37.20%) | `-0.0266` (-2.66%) |
| **Recall (Approved / Positive Class)** | `0.2727` (27.27%) | `0.4450` (44.50%) | **`+0.1723` (+17.23%)** |
| **F1-Score (Approved / Positive Class)** | `0.3239` (32.39%) | `0.4052` (40.52%) | **`+0.0813` (+8.13%)** |
| **ROC-AUC** | `0.5518` (55.18%) | `0.5422` (54.22%) | `-0.0096` (-0.96%) |
| **Average Precision (PR-AUC)** | `0.4126` (41.26%) | `0.3878` (38.78%) | `-0.0248` (-2.48%) |
| **No-Skill Positive Prevalence** | `0.3483` (34.83%) | `0.3483` (34.83%) | — |

---

## 3. Cross-Validation Stability Analysis (5-Fold Stratified CV on $X_{\text{train}}$, $N=2,400$)

To establish whether the holdout test set performance was an artifact of sampling variance, a 5-fold Stratified Cross-Validation protocol (`shuffle=True, random_state=42`) was conducted strictly on the 2,400 training samples:

| Metric | Random Forest (5-Fold Mean $\pm$ Std) | XGBoost (5-Fold Mean $\pm$ Std) | Holdout Test Benchmark | Cross-Validation Finding |
|---|---|---|---|---|
| **Accuracy** | $0.6042 \pm 0.0160$ | $0.5446 \pm 0.0136$ | RF: 0.6033 / XGB: 0.5450 | **Extremely stable across all folds** |
| **Precision** | $0.3991 \pm 0.0384$ | $0.3705 \pm 0.0159$ | RF: 0.3986 / XGB: 0.3720 | **High fold consistency** |
| **Recall** | $0.2739 \pm 0.0378$ | $0.4402 \pm 0.0299$ | RF: 0.2727 / XGB: 0.4450 | **Consistent recall elevation in XGBoost** |
| **F1-Score** | $0.3245 \pm 0.0385$ | $0.4021 \pm 0.0200$ | RF: 0.3239 / XGB: 0.4052 | **Consistent F1 advantage for XGBoost** |
| **ROC-AUC** | $0.5393 \pm 0.0200$ | $0.5269 \pm 0.0156$ | RF: 0.5518 / XGB: 0.5422 | **Stable ROC-AUC across training folds** |

### Stability Conclusion:
The standard deviations across all metrics are extremely low ($\sigma < 0.04$), confirming that the observed model performance is statistically stable and represents the intrinsic predictive capacity on this dataset representation, rather than random split variance.

---

## 4. Root-Cause Diagnostic: Distribution Shift & Feature-Target Decoupling

### 4.1 Target Distribution Shift
- **Empirical Seed Responses ($N=500$):** `Approved`: 232 (**46.40%**), `Rejected`: 268 (**53.60%**)
- **SDV CTGAN Synthetic Records ($N=2,500$):** `Approved`: 813 (**32.52%**), `Rejected`: 1,687 (**67.48%**)
- **Final Combined Benchmark ($N=3,000$):** `Approved`: 1,045 (**34.83%**), `Rejected`: 1,955 (**65.17%**)

**Finding:** The generative CTGAN synthesizer exhibited mode concentration toward the majority `Rejected` class, resulting in a **$-13.88\%$ percentage-point drop** in positive target prevalence.

---

### 4.2 Feature-Target Association Discrepancy (Empirical Seed vs. SDV Synthetic)

Statistical association between each predictor and `Loan Application Result` was quantified using the Chi-Square Test of Independence ($\chi^2$) and Cramér's V effect size:

$$\text{Cramér's } V = \sqrt{\frac{\chi^2}{N \cdot (\min(r, c) - 1)}}$$

| Feature Name | Empirical Seed ($N=500$) Cramér's V | Empirical Seed $p$-value | SDV Synthetic ($N=2500$) Cramér's V | SDV Synthetic $p$-value | Combined ($N=3000$) Cramér's V | Combined $p$-value |
|---|---|---|---|---|---|---|
| **`Monthly Income`** | **0.4106** | $< 0.000001$ | **0.0515** | $0.4696$ (NS) | **0.0854** | $0.0027$ |
| **`Requested Loan Amount`** | **0.3369** | $< 0.000001$ | **0.0564** | $0.4377$ (NS) | **0.0615** | $0.1831$ (NS) |
| **`Required Documents Submitted`** | **0.2980** | $< 0.000001$ | **0.0473** | $0.0613$ (NS) | **0.0806** | $0.0001$ |
| **`Employment Type`** | **0.2695** | $< 0.00001$ | **0.0192** | $0.9960$ (NS) | **0.0517** | $0.3310$ (NS) |
| **`Loan Type`** | **0.2666** | $< 0.00001$ | **0.0282** | $0.9208$ (NS) | **0.0476** | $0.3392$ (NS) |
| **`Loan Purpose`** | **0.2556** | $< 0.0001$ | **0.0680** | $0.1723$ (NS) | **0.0488** | $0.5209$ (NS) |
| **`Employment Duration`** | **0.2281** | $< 0.0001$ | **0.0486** | $0.3151$ (NS) | **0.0711** | $0.0097$ |
| **`Monthly Loan Repayment`** | **0.1587** | $0.0499$ | **0.0306** | $0.8863$ (NS) | **0.0154** | $0.9943$ (NS) |
| **`Collateral Availability`** | **0.1555** | $0.0024$ | **0.0127** | $0.8182$ (NS) | **0.0173** | $0.6400$ (NS) |
| **`Existing Loan Status`** | **0.1079** | $0.0158$ | **0.0081** | $0.6870$ (NS) | **0.0350** | $0.0552$ (NS) |
| **`Gender`** | **0.1188** | $0.0294$ | **0.0188** | $0.6443$ (NS) | **0.0087** | $0.8920$ (NS) |
| **`Education Level`** | 0.1172 | $0.3330$ (NS) | 0.0669 | $0.0829$ (NS) | 0.0612 | $0.0815$ (NS) |
| **`Marital Status`** | 0.1101 | $0.1943$ (NS) | 0.0330 | $0.6057$ (NS) | 0.0276 | $0.6849$ (NS) |
| **`Number of Dependents`** | 0.1129 | $0.2717$ (NS) | 0.0361 | $0.6611$ (NS) | 0.0258 | $0.8491$ (NS) |
| **`Age Group`** | 0.0754 | $0.7243$ (NS) | 0.0311 | $0.7899$ (NS) | 0.0195 | $0.9505$ (NS) |
| **`Guarantor Availability`** | 0.0525 | $0.2404$ (NS) | 0.0190 | $0.3410$ (NS) | 0.0088 | $0.6307$ (NS) |

*(Note: NS indicates Not Statistically Significant at $\alpha = 0.05$)*

### Critical Mathematical Finding:
1. **Strong Signal in Empirical Seed:** In the 500 Google Form responses, 10 out of 16 features exhibit statistically significant associations ($p < 0.05$) with loan approval, led by `Monthly Income` ($V=0.4106$), `Requested Loan Amount` ($V=0.3369$), and `Required Documents Submitted` ($V=0.2980$).
2. **Signal Decoupling in SDV CTGAN Synthesis:** In the 2,500 synthetic records, **zero** of the 16 features exhibit statistically significant association with the target (all $p > 0.05$, all $V < 0.07$). CTGAN successfully replicated marginal category frequencies (Column Shapes Score: 94.6%) but diluted the conditional joint distributions $P(Y \mid X_1, X_2, \dots, X_{16})$.
3. **Dilution Effect in Combined Dataset:** Because the synthetic records constitute 83.33% of the combined 3,000-record dataset, the strong underlying signal from the 500 seed records was substantially diluted, resulting in reduced classifier discriminability.

---

## 5. Diagnostic Visualizations

### 5.1 Receiver Operating Characteristic (ROC) Curve
![Receiver Operating Characteristic (ROC) Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/roc_curve.png)

- **Random Forest Baseline AUC:** `0.5518`
- **XGBoost Proposed Model AUC:** `0.5422`
- **Diagonal Reference (Random Chance):** `0.5000`
- **Interpretation:** Both models operate slightly above random guessing across varying classification thresholds, reflecting the diluted feature-target correlations in the 3,000-record benchmark.

---

### 5.2 Precision-Recall (PR) Curve
![Precision-Recall (PR) Curve](file:///C:/Users/User/.gemini/antigravity-ide/brain/053a421e-9739-4652-917c-e85f15169902/precision_recall_curve.png)

- **Random Forest Average Precision (AP):** `0.4126`
- **XGBoost Average Precision (AP):** `0.3878`
- **No-Skill Positive Prevalence:** `0.3483` (grey dashed line)
- **Interpretation:** Both models perform slightly above the baseline class prevalence line. XGBoost provides higher recall across low-to-mid precision operating points due to `scale_pos_weight=1.8708`.

---

## 6. Model Prediction Probability Distribution Diagnostic

Analyzing the continuous posterior probabilities $\hat{P}(\text{Approved}=1 \mid \mathbf{x})$ across the holdout test set ($N=600$):

| Model | Cohort | Sample Count | Mean Probability | Std Dev | Min | Median | Max | Probability Separation ($\Delta \mu$) |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | All Test Samples | 600 | 0.4382 | 0.0882 | 0.1850 | 0.4350 | 0.6950 | — |
| **Random Forest** | Actual Approved ($y=1$) | 209 | 0.4520 | 0.0863 | 0.2900 | 0.4450 | 0.6950 | \multirow{2}{*}{$\mathbf{\Delta \mu = 0.0212}$} |
| **Random Forest** | Actual Rejected ($y=0$) | 391 | 0.4308 | 0.0883 | 0.1850 | 0.4300 | 0.6950 | |
| **XGBoost** | All Test Samples | 600 | 0.4720 | 0.1185 | 0.0888 | 0.4707 | 0.8240 | — |
| **XGBoost** | Actual Approved ($y=1$) | 209 | 0.4860 | 0.1137 | 0.2032 | 0.4833 | 0.7962 | \multirow{2}{*}{$\mathbf{\Delta \mu = 0.0214}$} |
| **XGBoost** | Actual Rejected ($y=0$) | 391 | 0.4646 | 0.1203 | 0.0888 | 0.4686 | 0.8240 | |

### Probability Diagnostic Findings:
1. **Narrow Distribution Separation:** The difference between the mean predicted probability for actual Approved applicants vs. actual Rejected applicants is only $\approx 0.021$ (2.1 percentage points) for both models.
2. **Dense Probability Clustering:** In Random Forest, 50% of predictions fall tightly between $0.380$ and $0.495$. In XGBoost, 50% fall between $0.398$ and $0.558$.
3. **Threshold Sensitivity:** Because probabilities are clustered tightly around the decision boundary ($\approx 0.50$), small shifts in decision thresholds produce large swings in precision vs. recall.

---

## 7. Native Feature Importance Findings

### Top 10 Features: Random Forest (Gini Impurity) vs. XGBoost (Gain)

| Rank | Random Forest Feature | Gini Importance | XGBoost Feature | Gain Importance |
|---|---|---|---|---|
| 1 | `Requested Loan Amount` | 0.0710 (7.10%) | `Required Documents Submitted_Yes, all` | 0.0295 (2.95%) |
| 2 | `Monthly Income` | 0.0709 (7.09%) | `Loan Purpose_Education` | 0.0230 (2.30%) |
| 3 | `Employment Duration` | 0.0617 (6.17%) | `Loan Type_Education` | 0.0228 (2.28%) |
| 4 | `Number of Dependents` | 0.0571 (5.71%) | `Loan Purpose_Agriculture` | 0.0225 (2.25%) |
| 5 | `Age Group` | 0.0570 (5.70%) | `Guarantor Availability_Yes` | 0.0224 (2.24%) |
| 6 | `Monthly Loan Repayment` | 0.0475 (4.75%) | `Employment Type_Unemployed` | 0.0224 (2.24%) |
| 7 | `Collateral Availability_Yes` | 0.0197 (1.97%) | `Existing Loan Status_Yes` | 0.0218 (2.18%) |
| 8 | `Gender_Female` | 0.0192 (1.92%) | `Education Level_Primary` | 0.0210 (2.10%) |
| 9 | `Education Level_Undergraduate degree` | 0.0191 (1.91%) | `Loan Type_Vehicle` | 0.0209 (2.09%) |
| 10 | `Loan Type_Personal` | 0.0188 (1.88%) | `Employment Type_Government employee` | 0.0203 (2.03%) |

**Insight:** In Random Forest, continuous/ordinal features dominate Gini importance due to higher cardinality splits. In XGBoost, binary indicator features directly associated with credit policy (Document submission, Education/Agriculture loan type, Guarantor, Unemployment status) receive the highest split gain.

---

## 8. Limitations & Academic Assessment for Downstream Explainability

1. **Downstream Explainability Risk:** Explaining a machine learning model with an ROC-AUC of $\approx 0.54$ via SHAP and LIME means the extracted feature attributions will explain the internal logic of a relatively weak statistical estimator rather than true financial underwriting principles.
2. **Role of Agentic Policy Grounding:** This diagnostic underscores why a multi-layered **Agentic AI + RAG Architecture** is essential. A purely statistical ML model trained on unconstrained synthetic survey data cannot safely make lending decisions on its own; it requires regulatory constraint verification (e.g. debt-to-income caps, document verification rules) enforced by autonomous agents and policy retrieval.

---

## 9. Evidence-Based Decision Recommendation

Based on the diagnostic findings:

### Comparison of Paths:
- **Option A: Proceed directly to SHAP / LIME with the current XGBoost model.**
  - *Academic Rationale:* Accurately reflects the current state of the 3,000-record dataset pipeline, documents the exact fidelity trade-offs of CTGAN synthesis, and demonstrates how SHAP/LIME faithfully explain the actual fitted estimator (regardless of model strength).
- **Option B: Perform data synthesis / model enhancement before SHAP / LIME.**
  - *Academic Rationale:* Retune the SDV synthesizer with conditional sampling constraints (e.g. enforcing conditional target copulas or GaussianCopulaSynthesizer which showed higher pairwise correlation preservation of 73.74%), or apply hyperparameter optimization to improve predictive discrimination before generating explanations.

> **Diagnostic Recommendation:**
> Both options are mathematically valid for academic reporting. **Option B** is recommended if the primary goal is maximizing predictive benchmark metrics; **Option A** is recommended if the primary goal is demonstrating end-to-end pipeline interpretability under real-world synthetic data constraints.
