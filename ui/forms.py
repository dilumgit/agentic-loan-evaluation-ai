"""
Applicant Data Entry Forms and Validation Module.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Renders the 16-attribute manual applicant data entry form for Bank Credit Officers.
- Guarantees that selectable options match the exact categories in the frozen preprocessor.
- Validates that all 16 attributes are non-null and valid before execution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import streamlit as st


@st.cache_resource
def get_feature_categories() -> Dict[str, List[str]]:
    """Extract exact feature categories from the frozen preprocessor artifact."""
    workspace_root = Path(__file__).resolve().parent.parent
    preprocessor_path = workspace_root / "data" / "processed" / "preprocessor.joblib"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

    prep = joblib.load(preprocessor_path)

    ord_features = [
        "Age Group",
        "Number of Dependents",
        "Employment Duration",
        "Monthly Income",
        "Monthly Loan Repayment",
        "Requested Loan Amount",
    ]
    nom_features = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Employment Type",
        "Existing Loan Status",
        "Loan Type",
        "Loan Purpose",
        "Collateral Availability",
        "Guarantor Availability",
        "Required Documents Submitted",
    ]

    options: Dict[str, List[str]] = {}
    for feat, cats in zip(ord_features, prep.named_transformers_["ordinal"].categories_):
        options[feat] = list(cats)
    for feat, cats in zip(nom_features, prep.named_transformers_["nominal"].categories_):
        options[feat] = list(cats)

    return options


def get_sample_applicants() -> Dict[str, Dict[str, str]]:
    """Preset empirical applicant profiles for interactive testing."""
    categories = get_feature_categories()
    return {
        "-- Select Preset Profile (Optional) --": {},
        "Case 1 (True Positive): High Income, Complete Docs, Collateral Available": {
            "Age Group": categories["Age Group"][1],
            "Gender": categories["Gender"][2],
            "Marital Status": categories["Marital Status"][1],
            "Number of Dependents": categories["Number of Dependents"][3],
            "Education Level": categories["Education Level"][2],
            "Employment Type": categories["Employment Type"][4],
            "Employment Duration": categories["Employment Duration"][1],
            "Monthly Income": categories["Monthly Income"][4],
            "Existing Loan Status": "No",
            "Monthly Loan Repayment": categories["Monthly Loan Repayment"][0],
            "Loan Type": "Vehicle",
            "Requested Loan Amount": categories["Requested Loan Amount"][4],
            "Loan Purpose": "Other",
            "Collateral Availability": "Yes",
            "Guarantor Availability": "No",
            "Required Documents Submitted": "Yes, all",
        },
        "Case 2 (True Negative): Extreme Loan Request, Incomplete Docs, Contract Job": {
            "Age Group": categories["Age Group"][2],
            "Gender": "Female",
            "Marital Status": "Single",
            "Number of Dependents": "4",
            "Education Level": "Undergraduate degree",
            "Employment Type": "Contract employee",
            "Employment Duration": categories["Employment Duration"][2],
            "Monthly Income": categories["Monthly Income"][2],
            "Existing Loan Status": "No",
            "Monthly Loan Repayment": categories["Monthly Loan Repayment"][1],
            "Loan Type": "Business",
            "Requested Loan Amount": categories["Requested Loan Amount"][7],
            "Loan Purpose": "Agriculture",
            "Collateral Availability": "No",
            "Guarantor Availability": "No",
            "Required Documents Submitted": "No, some missing",
        },
        "Case 3 (False Positive Scenario): Education Loan, Clean History, Unsecured": {
            "Age Group": categories["Age Group"][1],
            "Gender": "Female",
            "Marital Status": "Single",
            "Number of Dependents": "0",
            "Education Level": "Secondary",
            "Employment Type": "Business owner",
            "Employment Duration": categories["Employment Duration"][3],
            "Monthly Income": categories["Monthly Income"][3],
            "Existing Loan Status": "No",
            "Monthly Loan Repayment": categories["Monthly Loan Repayment"][0],
            "Loan Type": "Education",
            "Requested Loan Amount": categories["Requested Loan Amount"][3],
            "Loan Purpose": "Debt consolidation",
            "Collateral Availability": "No",
            "Guarantor Availability": "Yes",
            "Required Documents Submitted": "Yes, all",
        },
        "Case 4 (False Negative Scenario): Low Income, Existing Loan, Missing Docs": {
            "Age Group": categories["Age Group"][2],
            "Gender": "Male",
            "Marital Status": "Single",
            "Number of Dependents": "1",
            "Education Level": "No formal education",
            "Employment Type": "Government employee",
            "Employment Duration": categories["Employment Duration"][4],
            "Monthly Income": categories["Monthly Income"][1],
            "Existing Loan Status": "Yes",
            "Monthly Loan Repayment": categories["Monthly Loan Repayment"][0],
            "Loan Type": "Housing",
            "Requested Loan Amount": categories["Requested Loan Amount"][4],
            "Loan Purpose": "House purchase/construction",
            "Collateral Availability": "Yes",
            "Guarantor Availability": "No",
            "Required Documents Submitted": "No, some missing",
        },
    }


def validate_applicant_data(data: Dict[str, Any], categories: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """Validate all 16 fields are present, non-empty, and belong to valid categories."""
    errors: List[str] = []
    expected_fields = [
        "Age Group",
        "Gender",
        "Marital Status",
        "Number of Dependents",
        "Education Level",
        "Employment Type",
        "Employment Duration",
        "Monthly Income",
        "Existing Loan Status",
        "Monthly Loan Repayment",
        "Loan Type",
        "Requested Loan Amount",
        "Loan Purpose",
        "Collateral Availability",
        "Guarantor Availability",
        "Required Documents Submitted",
    ]

    for field in expected_fields:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"Missing required field: '{field}'")
        elif field in categories and val not in categories[field]:
            errors.append(f"Invalid value '{val}' for field '{field}'. Allowed: {categories[field]}")

    return len(errors) == 0, errors


def render_applicant_form() -> Optional[Dict[str, Any]]:
    """Render the interactive manual entry form for bank officers."""
    categories = get_feature_categories()
    presets = get_sample_applicants()

    # Preset Profile Loader Selector
    selected_preset_name = st.selectbox(
        "Load Sample Applicant Profile (Optional Preset):",
        options=list(presets.keys()),
        index=0,
        help="Quickly populate the 16 fields with representative test profiles from Step 12.",
    )

    preset_data = presets.get(selected_preset_name, {})

    def get_val(key: str, default_idx: int = 0) -> int:
        if key in preset_data and preset_data[key] in categories[key]:
            return categories[key].index(preset_data[key])
        return min(default_idx, len(categories[key]) - 1)

    with st.form("applicant_evaluation_form", clear_on_submit=False):
        st.markdown("### 📋 1. Personal & Demographic Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_group = st.selectbox("Age Group *", options=categories["Age Group"], index=get_val("Age Group", 1))
            gender = st.selectbox("Gender *", options=categories["Gender"], index=get_val("Gender", 0))
        with c2:
            marital_status = st.selectbox("Marital Status *", options=categories["Marital Status"], index=get_val("Marital Status", 1))
            dependents = st.selectbox("Number of Dependents *", options=categories["Number of Dependents"], index=get_val("Number of Dependents", 2))
        with c3:
            education = st.selectbox("Education Level *", options=categories["Education Level"], index=get_val("Education Level", 6))

        st.markdown("### 💼 2. Employment & Cash-Flow Capacity")
        c4, c5, c6 = st.columns(3)
        with c4:
            employment_type = st.selectbox("Employment Type *", options=categories["Employment Type"], index=get_val("Employment Type", 4))
            employment_duration = st.selectbox("Employment Duration *", options=categories["Employment Duration"], index=get_val("Employment Duration", 3))
        with c5:
            monthly_income = st.selectbox("Monthly Income *", options=categories["Monthly Income"], index=get_val("Monthly Income", 3))
            existing_loans = st.selectbox("Existing Loan Status *", options=categories["Existing Loan Status"], index=get_val("Existing Loan Status", 0))
        with c6:
            monthly_repayment = st.selectbox("Monthly Loan Repayment *", options=categories["Monthly Loan Repayment"], index=get_val("Monthly Loan Repayment", 0))

        st.markdown("### 🏦 3. Loan Request & Security Details")
        c7, c8, c9 = st.columns(3)
        with c7:
            loan_type = st.selectbox("Loan Type *", options=categories["Loan Type"], index=get_val("Loan Type", 5))
            loan_amount = st.selectbox("Requested Loan Amount *", options=categories["Requested Loan Amount"], index=get_val("Requested Loan Amount", 2))
        with c8:
            loan_purpose = st.selectbox("Loan Purpose *", options=categories["Loan Purpose"], index=get_val("Loan Purpose", 7))
            collateral = st.selectbox("Collateral Availability *", options=categories["Collateral Availability"], index=get_val("Collateral Availability", 2))
        with c9:
            guarantor = st.selectbox("Guarantor Availability *", options=categories["Guarantor Availability"], index=get_val("Guarantor Availability", 0))
            docs_submitted = st.selectbox("Required Documents Submitted *", options=categories["Required Documents Submitted"], index=get_val("Required Documents Submitted", 2))

        st.markdown("---")
        submit_btn = st.form_submit_button("⚡ Evaluate Application (Run Multimodal Pipeline)", type="primary", use_container_width=True)

    if submit_btn:
        applicant_payload = {
            "Age Group": age_group,
            "Gender": gender,
            "Marital Status": marital_status,
            "Number of Dependents": dependents,
            "Education Level": education,
            "Employment Type": employment_type,
            "Employment Duration": employment_duration,
            "Monthly Income": monthly_income,
            "Existing Loan Status": existing_loans,
            "Monthly Loan Repayment": monthly_repayment,
            "Loan Type": loan_type,
            "Requested Loan Amount": loan_amount,
            "Loan Purpose": loan_purpose,
            "Collateral Availability": collateral,
            "Guarantor Availability": guarantor,
            "Required Documents Submitted": docs_submitted,
        }

        is_valid, errs = validate_applicant_data(applicant_payload, categories)
        if not is_valid:
            for err in errs:
                st.error(f"❌ {err}")
            return None

        return applicant_payload

    return None
