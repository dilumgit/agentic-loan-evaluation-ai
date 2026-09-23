"""
UI Helper Package for Streamlit Decision Support Interface.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

from ui.styles import apply_custom_styles
from ui.forms import render_applicant_form, get_feature_categories, get_sample_applicants
from ui.components import (
    render_header_and_disclaimer,
    render_ml_prediction_section,
    render_xai_section,
    render_policy_rag_section,
    render_decision_agent_section,
    render_audit_trail_section,
)
from ui.policy_ui import render_policy_management_page, get_rag_components

__all__ = [
    "apply_custom_styles",
    "render_applicant_form",
    "get_feature_categories",
    "get_sample_applicants",
    "render_header_and_disclaimer",
    "render_ml_prediction_section",
    "render_xai_section",
    "render_policy_rag_section",
    "render_decision_agent_section",
    "render_audit_trail_section",
    "render_policy_management_page",
    "get_rag_components",
]
