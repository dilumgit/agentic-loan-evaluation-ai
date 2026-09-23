"""
Streamlit Decision Support Interface for Bank Loan Evaluation.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Application Purpose:
- Interactive web-based decision support prototype for Bank Credit Officers.
- Unifies Manual Applicant Data Entry, Preprocessing, Frozen XGBoost Model Scoring,
  SHAP/LIME Explainability, Policy RAG, Central Groq AI Decision Agent, and Policy Management.
"""

from pathlib import Path
import sys
from typing import Optional

import streamlit as st

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agents.llm_service import GroqLLMService
from agents.orchestrator import LoanEvaluationOrchestrator
from ui.components import (
    render_audit_trail_section,
    render_decision_agent_section,
    render_header_and_disclaimer,
    render_ml_prediction_section,
    render_policy_rag_section,
    render_xai_section,
)
from ui.forms import render_applicant_form
from ui.policy_ui import render_policy_management_page
from ui.styles import apply_custom_styles


@st.cache_resource
def load_orchestrator() -> LoanEvaluationOrchestrator:
    """Load and cache the master loan evaluation orchestrator."""
    return LoanEvaluationOrchestrator()


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="Bank Loan Decision Support System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Apply Custom Styling
    apply_custom_styles()

    # 3. Sidebar: Navigation & Diagnostics
    with st.sidebar:
        st.markdown("### 🏛️ System Navigation")
        nav_mode = st.radio(
            "Select Interface:",
            options=["🏦 Loan Evaluation", "📚 Policy Management"],
            index=0,
            help="Switch between Loan Applicant Evaluation and Bank Policy Knowledge Base Management.",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Component Status")
        st.markdown("✅ **Dataset:** `Dataset v2 (3,000 records)`")
        st.markdown("✅ **Model:** `XGBoost (AUC: 64.68%)`")
        st.markdown("✅ **XAI Layer:** `TreeSHAP & LIME`")
        st.markdown("✅ **RAG Engine:** `ChromaDB Vector Store`")

        llm_check = GroqLLMService()
        if llm_check.is_available():
            st.markdown(f"✅ **Groq LLM:** `Active ({llm_check.model_name})`")
        else:
            st.markdown("ℹ️ **Groq LLM:** `Offline / Fallback Engine`")

        st.markdown("---")
        st.markdown("### 🔄 Application Control")
        if st.button("🧹 Clear / New Application", use_container_width=True):
            st.session_state.pop("applicant_data", None)
            st.session_state.pop("decision_result", None)
            st.session_state.pop("audit_record", None)
            st.session_state.pop("evaluation_complete", None)
            st.rerun()

        st.caption("Developed for Academic Research in Explainable & Policy-Aware Agentic AI Decision Support.")

    # 4. Route Based on Selected Navigation Mode
    if nav_mode == "📚 Policy Management":
        render_policy_management_page()
        return

    # -------------------------------------------------------------------------
    # LOAN EVALUATION WORKFLOW
    # -------------------------------------------------------------------------
    # 5. Main Header & Regulatory Notice
    render_header_and_disclaimer()

    # 6. Load Master Orchestrator
    try:
        orchestrator = load_orchestrator()
    except Exception as e:
        st.error(f"❌ Failed to initialize loan evaluation engine: {str(e)}")
        st.stop()

    # 7. Check Session State
    if "evaluation_complete" not in st.session_state:
        st.session_state["evaluation_complete"] = False

    # 8. Applicant Data Entry Form
    applicant_payload = render_applicant_form()

    if applicant_payload:
        with st.spinner("⏳ Running multimodal evaluation pipeline (ML Scoring → TreeSHAP → LIME → Policy RAG → Decision Agent)..."):
            try:
                decision_result, audit_record = orchestrator.evaluate_application(applicant_payload)
                st.session_state["applicant_data"] = applicant_payload
                st.session_state["decision_result"] = decision_result
                st.session_state["audit_record"] = audit_record
                st.session_state["evaluation_complete"] = True
            except Exception as e:
                st.error(f"❌ Application evaluation error: {str(e)}")
                st.session_state["evaluation_complete"] = False

    # 9. Render Results View if Evaluation is Complete
    if st.session_state.get("evaluation_complete") and "decision_result" in st.session_state:
        decision_result = st.session_state["decision_result"]
        audit_record = st.session_state["audit_record"]

        st.markdown("---")
        st.markdown("## 📑 Evaluation Results & Multimodal Decision Support")

        # Layout Section Cards
        render_ml_prediction_section(decision_result)
        st.markdown("---")
        render_xai_section(decision_result, audit_record)
        st.markdown("---")
        render_policy_rag_section(decision_result, audit_record)
        st.markdown("---")
        render_decision_agent_section(decision_result)
        st.markdown("---")
        render_audit_trail_section(audit_record)


if __name__ == "__main__":
    main()
