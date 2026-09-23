"""
Central Orchestration Pipeline for Policy-Aware Loan Decision Support System.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Integrates Preprocessing, Frozen XGBoost scoring, SHAP/LIME XAI attributions, Policy RAG retrieval,
  and the Central AI Decision Agent.
- Generates comprehensive immutable AuditTrailRecords.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agents.decision_agent import CentralPolicyAwareDecisionAgent
from agents.llm_service import GroqLLMService
from agents.schemas import (
    AuditTrailRecord,
    DecisionAgentInput,
    DecisionSupportResult,
)
from rag.config import default_rag_config
from rag.retriever import PolicyRetriever


class LoanEvaluationOrchestrator:
    """Master orchestrator integrating ML scoring, XAI explanations, RAG policy retrieval, and Central AI reasoning."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None,
        retriever: Optional[PolicyRetriever] = None,
        decision_agent: Optional[CentralPolicyAwareDecisionAgent] = None,
    ):
        self.workspace_root = WORKSPACE_ROOT
        data_processed_dir = self.workspace_root / "data" / "processed"
        models_dir = self.workspace_root / "models" / "xgboost"

        # 1. Load Frozen Preprocessing & Model Artifacts
        self.model_path = model_path or (models_dir / "final_xgboost.joblib")
        self.preprocessor_path = preprocessor_path or (data_processed_dir / "preprocessor.joblib")
        self.feature_names_path = feature_names_path or (data_processed_dir / "feature_names.json")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_names_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        # 2. Initialize SHAP Explainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # 3. Initialize LIME Explainer
        X_train_df = pd.read_csv(data_processed_dir / "X_train.csv")
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=self.feature_names,
            class_names=["Rejected (0)", "Approved (1)"],
            mode="classification",
            random_state=42,
            discretize_continuous=True,
        )

        # 4. Initialize Policy RAG Retriever & Decision Agent
        self.retriever = retriever or PolicyRetriever(default_rag_config)
        self.decision_agent = decision_agent or CentralPolicyAwareDecisionAgent()

    def construct_policy_query(self, applicant_data: Dict[str, Any]) -> str:
        """Dynamically generate a contextual policy retrieval query from applicant attributes."""
        loan_type = applicant_data.get("Loan Type", "General")
        amount = applicant_data.get("Requested Loan Amount", "Standard")
        collateral = applicant_data.get("Collateral Availability", "Not specified")
        employment = applicant_data.get("Employment Type", "General")
        purpose = applicant_data.get("Loan Purpose", "Personal")

        query = (
            f"Underwriting policies for {loan_type} loan requested amount {amount} "
            f"purpose {purpose} employment {employment} collateral requirement {collateral}"
        )
        return query

    def evaluate_application(
        self,
        applicant_data: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Tuple[DecisionSupportResult, AuditTrailRecord]:
        """Execute end-to-end evaluation: ML -> SHAP/LIME -> RAG -> Central AI Agent."""
        req_id = request_id or f"REQ_{uuid.uuid4().hex[:10].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Transform raw applicant record into 55-dimensional feature matrix
        df_raw = pd.DataFrame([applicant_data])
        X_transformed = self.preprocessor.transform(df_raw)
        X_transformed_df = pd.DataFrame(X_transformed, columns=self.feature_names)

        # 2. Compute Frozen XGBoost Predictions
        pred_label_idx = int(self.model.predict(X_transformed)[0])
        probas = self.model.predict_proba(X_transformed)[0]
        rejection_prob = float(probas[0])
        approval_prob = float(probas[1])
        ml_prediction = "Approved" if pred_label_idx == 1 else "Rejected"

        # 3. Compute Local SHAP Attributions
        shap_vals = self.shap_explainer.shap_values(X_transformed_df)[0]
        df_shap = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Value": shap_vals,
        })
        top_shap_pos = df_shap.sort_values(by="SHAP_Value", ascending=False).head(3)
        top_shap_neg = df_shap.sort_values(by="SHAP_Value", ascending=True).head(3)

        shap_pos_factors = [f"{r['Feature']} (+{r['SHAP_Value']:.3f})" for _, r in top_shap_pos.iterrows()]
        shap_neg_factors = [f"{r['Feature']} ({r['SHAP_Value']:.3f})" for _, r in top_shap_neg.iterrows()]

        # 4. Compute Local LIME Attributions
        lime_exp = self.lime_explainer.explain_instance(
            data_row=X_transformed[0],
            predict_fn=self.model.predict_proba,
            num_features=8,
            labels=(1,),
        )
        lime_list = lime_exp.as_list(label=1)
        lime_pos_factors = [f"{feat} (+{weight:.3f})" for feat, weight in lime_list if weight > 0][:3]
        lime_neg_factors = [f"{feat} ({weight:.3f})" for feat, weight in lime_list if weight < 0][:3]

        # 5. Construct Policy Retrieval Query & Retrieve RAG Evidence
        policy_query = self.construct_policy_query(applicant_data)
        retrieved_evidence = self.retriever.retrieve(query=policy_query, top_k=3, filter_active_only=True)
        policy_status_ctx = "POLICY_DOCUMENTS_ACTIVE" if retrieved_evidence else "NO_POLICY_AVAILABLE"

        # 6. Assemble Agent Input Payload
        agent_input = DecisionAgentInput(
            request_id=req_id,
            timestamp=timestamp,
            applicant_data=applicant_data,
            ml_prediction=ml_prediction,
            approval_probability=round(approval_prob, 4),
            rejection_probability=round(rejection_prob, 4),
            shap_top_positive_factors=shap_pos_factors,
            shap_top_negative_factors=shap_neg_factors,
            lime_top_positive_factors=lime_pos_factors,
            lime_top_negative_factors=lime_neg_factors,
            policy_retrieval_query=policy_query,
            retrieved_policy_evidence=retrieved_evidence,
            policy_status_context=policy_status_ctx,
        )

        # 7. Execute Central AI Decision Agent Reasoning
        decision_result = self.decision_agent.reason(agent_input)

        # 8. Construct Immutable Audit Trail Record
        audit_record = AuditTrailRecord(
            request_id=req_id,
            timestamp=timestamp,
            model_name="XGBoost Classifier (models/xgboost/final_xgboost.joblib)",
            model_version="Dataset_v2_GaussianCopula_3000",
            applicant_data=applicant_data,
            ml_prediction=ml_prediction,
            approval_probability=round(approval_prob, 4),
            rejection_probability=round(rejection_prob, 4),
            shap_top_positive=shap_pos_factors,
            shap_top_negative=shap_neg_factors,
            lime_top_positive=lime_pos_factors,
            lime_top_negative=lime_neg_factors,
            rag_query=policy_query,
            retrieved_chunks_count=len(retrieved_evidence),
            retrieved_chunks_summary=[
                {
                    "doc": item.get("document_name"),
                    "page": item.get("page_number"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("relevance_score"),
                }
                for item in retrieved_evidence
            ],
            policy_status=decision_result.policy_status.value,
            final_recommendation=decision_result.final_recommendation.value,
            reasoning_summary=decision_result.reasoning_summary,
            confidence=decision_result.confidence,
        )

        return decision_result, audit_record
