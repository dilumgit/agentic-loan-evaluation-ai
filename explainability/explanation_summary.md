# Explainable AI (XAI) Architecture & Evaluation Summary

## 1. Research Overview & Objectives
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 5 — Explainable AI Integration (SHAP & LIME)**
- **Objective:** Provide mathematically grounded, transparent explanations for the predictive behavior of the proposed XGBoost loan classification model. Explanations operate at two levels:
  1. **Global Explanations:** Explaining overall model decision dynamics across the entire test population.
  2. **Local Explanations:** Explaining feature contributions for specific, individual loan applicant instances.

---

## 2. Theoretical Foundations: SHAP vs. LIME

| Dimension | SHAP (SHapley Additive exPlanations) | LIME (Local Interpretable Model-agnostic Explanations) |
|---|---|---|
| **Theoretical Foundation** | Cooperative Game Theory (Shapley Values) | Local Linear Surrogate Modeling |
| **Explanation Scope** | Global (entire dataset) & Local (individual sample) | Local (individual sample) |
| **Consistency Property** | **Guaranteed:** If a model changes so that a feature contributes more, its attribution will never decrease (Efficiency, Symmetry, Dummy, Additivity). | **Heuristic:** May vary slightly across perturbation samplings due to local Monte Carlo approximation. |
| **Underlying Algorithm** | TreeSHAP (polynomial-time exact computation on tree ensemble graph) | Local perturbation with exponential kernel weighting and Ridge regression |
| **Output Representation** | Additive log-odds / probability impact ($f(x) = \phi_0 + \sum_{i=1}^M \phi_i$) | Linear regression coefficients on local feature conditions |

---

## 3. Dataset & Model Context
- **Model Used:** Trained `XGBClassifier` (`models/xgboost/xgboost_model.joblib`)
- **Evaluation Dataset:** Holdout test matrix (`data/processed/X_test.csv`, $N = 400$, 55 features)
- **Target Classes:** `Approved` ($1$), `Rejected` ($0$)

---

## 4. Global Explanations (SHAP Findings)

Across the 400 test applicants, the top 10 most influential features driving XGBoost predictions globally are:

| Rank | Feature Name | Mean Absolute SHAP Value | Global Directional Impact |
|---|---|---|---|
| **1** | `Monthly Income` | **`0.9268`** | Higher monthly income strongly increases probability of approval. |
| **2** | `Requested Loan Amount` | **`0.7459`** | Disproportionately large requested amounts strongly increase rejection likelihood. |
| **3** | `Required Documents Submitted_Yes, all` | **`0.3867`** | Full submission of mandatory paperwork significantly boosts approval log-odds. |
| **4** | `Monthly Loan Repayment` | **`0.3417`** | High existing debt obligations decrease approval probability. |
| **5** | `Collateral Availability_Yes` | **`0.2682`** | Offering eligible collateral positively supports approval. |
| **6** | `Employment Duration` | **`0.1934`** | Longer continuous employment tenure provides positive stability contribution. |
| **7** | `Collateral Availability_No` | **`0.1379`** | Absence of collateral increases probability of rejection. |
| **8** | `Guarantor Availability_No` | **`0.1207`** | Lack of a third-party guarantor contributes negatively to high-risk applicants. |
| **9** | `Employment Type_Unemployed` | **`0.1043`** | Unemployed status serves as a prominent negative contributor. |
| **10** | `Number of Dependents` | **`0.0833`** | High dependent counts moderately increase financial strain and rejection risk. |

### Visual Artifacts Generated:
- **`explainability/shap_summary.png`:** Global beeswarm summary plot illustrating distribution of SHAP values across feature ranges.
- **`explainability/shap_feature_importance.png`:** Bar chart ranking mean absolute SHAP values.

---

## 5. Local Explanations (Sample Instance Analysis)

Local explanations were generated for 5 test records ($i = 0, 1, 2, 3, 4$). 

### Example: Test Sample #0
- **Actual Label:** `Rejected (0)`
- **XGBoost Predicted Label:** `Rejected (0)`
- **Predicted Probability of Approval:** `0.0201` ($2.01\%$)

#### SHAP Local Attribution (Sample #0):
- Base Value (Dataset Average Expected Log-Odds): `0.0076`
- **Primary Negative Contributors (Pushing towards Rejection):**
  - `Required Documents Submitted_Yes, all = 0.0` ($\phi = -1.3157$)
  - `Monthly Loan Repayment = 5.0 (Rs.75,000–99,999)` ($\phi = -0.8789$)
  - `Monthly Income = 1.0 (Rs.50,000–99,999)` ($\phi = -0.7925$)
  - `Required Documents Submitted_No, some missing = 1.0` ($\phi = -0.5686$)
  - `Requested Loan Amount = 3.0 (Rs.5,000,000–9,999,999)` ($\phi = -0.3073$)
- **Primary Positive Contributors:**
  - `Collateral Availability_Yes = 1.0` ($\phi = +0.2604$)
  - `Employment Type_Retired = 1.0` ($\phi = +0.2152$)

#### LIME Local Attribution (Sample #0):
- **Surrogate Weights (Top Influences on Class 1 / Approval):**
  - `Monthly Income <= 2.00`: Weight = `-0.2413`
  - `2.00 < Requested Loan Amount <= 4.00`: Weight = `-0.1016`
  - `Required Documents Submitted_No, some missing > 0.00`: Weight = `-0.0994`
  - `0.00 < Collateral Availability_Yes <= 1.00`: Weight = `+0.0960`
  - `Monthly Loan Repayment > 2.00`: Weight = `-0.0855`

Both explainers exhibit strong convergence: low income, high existing debt obligations, high requested loan amount, and incomplete documentation are identified as the primary factors that **contributed to the model's rejection prediction**.

---

## 6. Critical Academic & Methodological Distinctions

1. **Statistical Contribution vs. Causality:**
   - SHAP and LIME values quantify how specific feature values *contributed to the machine learning model's output*, not physical causality in the real world.
   - Language used in reports and downstream agents explicitly states that a feature *"contributed to the model prediction"* rather than *"caused the loan outcome."*

2. **Model Prediction vs. Policy Rules:**
   - Machine learning predictions and XAI attributions are strictly statistical signals.
   - They do **not** constitute official bank lending policy rules.
   - Policy grounding, compliance verification, and statutory constraints are handled by the dedicated **RAG Policy Knowledge Store** and **Policy Reasoning Agents** in subsequent phases.

3. **Decision Support Hierarchy:**
   $$\text{Raw Application} \xrightarrow{\text{Preprocess}} \text{XGBoost Prediction} \xrightarrow{\text{XAI}} \text{SHAP/LIME Explanations} \xrightarrow{\text{RAG + Agents}} \text{Actionable Recommendation for Human Loan Officer}$$

---

## 7. Limitations of SHAP and LIME

### 7.1 Limitations of SHAP
- **Computational Sensitivity to Correlated Features:** Although TreeSHAP handles tree graph structures efficiently, strong feature correlations can sometimes distribute attribution among collinear features.
- **Log-Odds vs. Probability Scale:** TreeSHAP default outputs operate in margin (log-odds) space; non-linear inverse-logit transformations are required when converting directly to probability increments.

### 7.2 Limitations of LIME
- **Sampling Instability:** Because LIME fits local linear models on randomly perturbed neighborhood samples, explanation weights can exhibit slight stochastic variation between independent runs if random seeds are unpinned.
- **Hyperparameter Sensitivity:** Explanations vary depending on the perturbation kernel width and the number of features selected for the surrogate model.
