# Feature Mapping & Interpretation Document

## 1. Context & Architecture
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Implementation Phase:** **Step 5 — Explainable AI (SHAP & LIME)**
- **Purpose:** Provide an exact reference mapping between the **16 original raw loan features** and the **55 preprocessed features** utilized by the XGBoost classifier, SHAP TreeExplainer, and LIME tabular explainer.

---

## 2. Feature Mapping Matrix

| Original Feature (#) | Original Domain Type | Transformed Feature Name(s) in XGBoost / SHAP / LIME | Encoding Strategy | Representation & Value Space |
|---|---|---|---|---|
| **1. Age Group** | Categorical (Age bracket) | `Age Group` | Ordinal | Monotonic Integer (`0` to `5`): `Below 25` (0), `25–34` (1), `35–44` (2), `45–54` (3), `55–64` (4), `65 or above` (5) |
| **2. Gender** | Categorical (Nominal) | `Gender_Female`<br>`Gender_Male`<br>`Gender_Prefer not to say` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **3. Marital Status** | Categorical (Nominal) | `Marital Status_Divorced`<br>`Marital Status_Married`<br>`Marital Status_Prefer not to say`<br>`Marital Status_Single`<br>`Marital Status_Widowed` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **4. Number of Dependents** | Categorical / Count | `Number of Dependents` | Ordinal | Monotonic Integer (`0` to `5`): `0` (0), `1` (1), `2` (2), `3` (3), `4` (4), `5 or more` (5) |
| **5. Education Level** | Categorical (Nominal) | `Education Level_Diploma`<br>`Education Level_No formal education`<br>`Education Level_Other`<br>`Education Level_Postgraduate degree`<br>`Education Level_Primary`<br>`Education Level_Secondary`<br>`Education Level_Undergraduate degree` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **6. Employment Type** | Categorical (Nominal) | `Employment Type_Business owner`<br>`Employment Type_Contract employee`<br>`Employment Type_Government employee`<br>`Employment Type_Other`<br>`Employment Type_Private-sector employee`<br>`Employment Type_Retired`<br>`Employment Type_Self-employed`<br>`Employment Type_Unemployed` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **7. Employment Duration** | Categorical (Duration bracket) | `Employment Duration` | Ordinal | Monotonic Integer (`0` to `5`): `Not employed/business` (0), `Less than 1 year` (1), `1–2 years` (2), `3–5 years` (3), `6–10 years` (4), `More than 10 years` (5) |
| **8. Monthly Income** | Categorical (Income bracket) | `Monthly Income` | Ordinal | Monotonic Integer (`0` to `7`): `Below Rs.50,000` (0) to `Rs.1,000,000 or above` (7) |
| **9. Existing Loan Status** | Categorical (Binary status) | `Existing Loan Status_No`<br>`Existing Loan Status_Yes` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **10. Monthly Loan Repayment** | Categorical (Repayment bracket) | `Monthly Loan Repayment` | Ordinal | Monotonic Integer (`0` to `6`): `No existing loan` (0) to `Rs.100,000+` (6) |
| **11. Loan Type** | Categorical (Nominal) | `Loan Type_Agricultural`<br>`Loan Type_Business`<br>`Loan Type_Education`<br>`Loan Type_Housing`<br>`Loan Type_Other`<br>`Loan Type_Personal`<br>`Loan Type_Vehicle` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **12. Requested Loan Amount** | Categorical (Amount bracket) | `Requested Loan Amount` | Ordinal | Monotonic Integer (`0` to `8`): `Below Rs.500,000` (0) to `Rs.100,000,000 or above` (8) |
| **13. Loan Purpose** | Categorical (Nominal) | `Loan Purpose_Agriculture`<br>`Loan Purpose_Business investment`<br>`Loan Purpose_Debt consolidation`<br>`Loan Purpose_Education`<br>`Loan Purpose_House purchase/construction`<br>`Loan Purpose_Medical expenses`<br>`Loan Purpose_Other`<br>`Loan Purpose_Personal expenses`<br>`Loan Purpose_Vehicle purchase` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **14. Collateral Availability** | Categorical (Nominal) | `Collateral Availability_No`<br>`Collateral Availability_Not applicable / Not required`<br>`Collateral Availability_Yes` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **15. Guarantor Availability** | Categorical (Nominal) | `Guarantor Availability_No`<br>`Guarantor Availability_Yes` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |
| **16. Required Documents Submitted** | Categorical (Nominal) | `Required Documents Submitted_No, some missing`<br>`Required Documents Submitted_Not sure`<br>`Required Documents Submitted_Yes, all` | One-Hot | Binary Indicator (`1.0` if active, `0.0` otherwise) |

---

## 3. Preservation of Explainability Semantics

1. **Direct Attribution:** In SHAP and LIME, one-hot encoded features represent the exact presence (`1.0`) or absence (`0.0`) of a specific category (e.g. `Collateral Availability_Yes = 1.0` vs `Collateral Availability_No = 1.0`).
2. **Monotonic Progression:** Ordinal features (e.g., `Monthly Income`, `Requested Loan Amount`) maintain integer index ranks, allowing tree split criteria to directly correlate with higher/lower financial capacity.
3. **No Unseen Renaming:** All downstream XAI modules and agents access features strictly via `data/processed/feature_names.json`.
