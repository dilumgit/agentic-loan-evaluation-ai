"""
Structured Schemas and Pydantic Models for Central AI Decision Agent.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Defines strongly-typed Pydantic schemas for agent inputs, structured outputs, evidence citations,
  and audit trail logging.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyStatus(str, Enum):
    """Categorical evaluation of policy alignment."""
    POLICY_SUPPORTED = "POLICY_SUPPORTED"
    POLICY_CONFLICT = "POLICY_CONFLICT"
    INSUFFICIENT_POLICY_EVIDENCE = "INSUFFICIENT_POLICY_EVIDENCE"
    NO_POLICY_AVAILABLE = "NO_POLICY_AVAILABLE"


class DecisionRecommendation(str, Enum):
    """Controlled decision-support recommendation categories."""
    APPROVE_SUPPORT = "APPROVE_SUPPORT"
    REJECT_SUPPORT = "REJECT_SUPPORT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class EvidenceCitation(BaseModel):
    """Citation metadata for retrieved policy clauses."""
    document_name: str = Field(..., description="Name of the source bank policy document")
    page_number: int = Field(default=-1, description="Source page number where available, or -1")
    chunk_id: str = Field(..., description="Unique chunk identifier in vector database")
    relevance_score: float = Field(default=0.0, description="Vector similarity relevance score")
    clause_snippet: str = Field(default="", description="Explanatory excerpt from the retrieved policy clause")


class DecisionAgentInput(BaseModel):
    """Validated input payload presented to the Central AI Decision Agent."""
    request_id: str = Field(..., description="Unique evaluation request identifier")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")
    applicant_data: Dict[str, Any] = Field(..., description="Raw 16-attribute applicant dictionary")
    
    # Statistical Machine Learning Core (FROZEN)
    ml_prediction: str = Field(..., description="Raw XGBoost prediction: 'Approved' or 'Rejected'")
    approval_probability: float = Field(..., ge=0.0, le=1.0, description="XGBoost P(Approved)")
    rejection_probability: float = Field(..., ge=0.0, le=1.0, description="XGBoost P(Rejected)")
    
    # Explainable AI Attributions
    shap_top_positive_factors: List[str] = Field(default_factory=list, description="Top positive factors from SHAP")
    shap_top_negative_factors: List[str] = Field(default_factory=list, description="Top negative factors from SHAP")
    lime_top_positive_factors: List[str] = Field(default_factory=list, description="Top positive factors from LIME")
    lime_top_negative_factors: List[str] = Field(default_factory=list, description="Top negative factors from LIME")
    
    # Policy RAG Context
    policy_retrieval_query: str = Field(..., description="Dynamically generated policy search query")
    retrieved_policy_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved policy clauses")
    policy_status_context: str = Field(default="NO_POLICY_AVAILABLE", description="Pre-evaluation RAG repository state")


class DecisionSupportResult(BaseModel):
    """Strictly structured output schema produced by the Central AI Decision Agent."""
    request_id: str = Field(..., description="Evaluation request identifier")
    timestamp: str = Field(..., description="Timestamp of recommendation generation")
    
    # Preserved Statistical Core (MUST NOT BE ALTERED)
    ml_prediction: str = Field(..., description="Preserved XGBoost prediction")
    approval_probability: float = Field(..., description="Preserved XGBoost approval probability")
    rejection_probability: float = Field(..., description="Preserved XGBoost rejection probability")
    
    # Extracted Model & Evidence Factors
    key_positive_factors: List[str] = Field(..., description="Key factors supporting creditworthiness")
    key_negative_factors: List[str] = Field(..., description="Key risk factors weighing against approval")
    
    # Policy Reasoning Layer
    policy_status: PolicyStatus = Field(..., description="Policy compliance status")
    policy_findings: List[str] = Field(default_factory=list, description="Identified policy alignment findings")
    policy_conflicts: List[str] = Field(default_factory=list, description="Identified policy violations or discrepancies")
    missing_information: List[str] = Field(default_factory=list, description="Missing documentation or data fields")
    
    # Decision Support Recommendation
    reasoning_summary: str = Field(..., description="Comprehensive synthesis explaining the recommendation")
    final_recommendation: DecisionRecommendation = Field(..., description="Controlled decision-support recommendation")
    confidence: float = Field(default=0.75, ge=0.0, le=1.0, description="Agent confidence in the recommendation")
    
    # Evidence & Disclaimers
    evidence_citations: List[EvidenceCitation] = Field(default_factory=list, description="Cited policy evidence clauses")
    disclaimer: str = Field(
        default="NOTICE: This recommendation is generated by an AI Decision Support System and does not constitute a final banking approval or commitment. Final underwriting decisions rest solely with authorized human bank credit officers.",
        description="Mandatory regulatory decision-support disclaimer"
    )


class AuditTrailRecord(BaseModel):
    """Comprehensive immutable audit log record for underwriting governance."""
    request_id: str
    timestamp: str
    model_name: str = "XGBoost (final_xgboost.joblib)"
    model_version: str = "Dataset_v2_Holdout_AUC_0.6468"
    applicant_data: Dict[str, Any]
    ml_prediction: str
    approval_probability: float
    rejection_probability: float
    shap_top_positive: List[str]
    shap_top_negative: List[str]
    lime_top_positive: List[str]
    lime_top_negative: List[str]
    rag_query: str
    retrieved_chunks_count: int
    retrieved_chunks_summary: List[Dict[str, Any]]
    policy_status: str
    final_recommendation: str
    reasoning_summary: str
    confidence: float
