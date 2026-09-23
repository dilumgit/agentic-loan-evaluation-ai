"""
Integration Tests for Loan Evaluation Orchestrator Pipeline.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

from pathlib import Path
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from agents.orchestrator import LoanEvaluationOrchestrator
from agents.schemas import AuditTrailRecord, DecisionSupportResult, PolicyStatus


class TestOrchestrator(unittest.TestCase):
    """Integration test suite for the full ML -> XAI -> RAG -> Decision Agent pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = LoanEvaluationOrchestrator()

    def test_full_evaluation_pipeline(self):
        """Test complete orchestration on a sample test applicant."""
        applicant_data = {
            "Age Group": "25–34",
            "Gender": "Female",
            "Marital Status": "Married",
            "Number of Dependents": "2",
            "Education Level": "Postgraduate degree",
            "Employment Type": "Self-employed",
            "Employment Duration": "6–10 years",
            "Monthly Income": "Rs.150,000–249,999",
            "Existing Loan Status": "No",
            "Monthly Loan Repayment": "No existing loan",
            "Loan Type": "Vehicle",
            "Requested Loan Amount": "Rs.1,000,000–4,999,999",
            "Loan Purpose": "Vehicle purchase",
            "Collateral Availability": "Yes",
            "Guarantor Availability": "No",
            "Required Documents Submitted": "Yes, all",
        }

        decision_result, audit_record = self.orchestrator.evaluate_application(
            applicant_data=applicant_data,
            request_id="REQ_INTEGRATION_TEST_001",
        )

        # 1. Verify types
        self.assertIsInstance(decision_result, DecisionSupportResult)
        self.assertIsInstance(audit_record, AuditTrailRecord)

        # 2. Verify ML prediction preservation
        self.assertIn(decision_result.ml_prediction, ["Approved", "Rejected"])
        self.assertGreaterEqual(decision_result.approval_probability, 0.0)
        self.assertLessEqual(decision_result.approval_probability, 1.0)
        self.assertAlmostEqual(
            decision_result.approval_probability + decision_result.rejection_probability,
            1.0,
            places=3,
        )

        # 3. Verify Explainable AI factors
        self.assertGreater(len(decision_result.key_positive_factors) + len(decision_result.key_negative_factors), 0)

        # 4. Verify Policy RAG Integration and Policy Status
        self.assertIsInstance(decision_result.policy_status, PolicyStatus)
        self.assertIn(
            decision_result.policy_status,
            [
                PolicyStatus.POLICY_SUPPORTED,
                PolicyStatus.POLICY_CONFLICT,
                PolicyStatus.INSUFFICIENT_POLICY_EVIDENCE,
                PolicyStatus.NO_POLICY_AVAILABLE,
            ],
        )
        self.assertGreater(audit_record.retrieved_chunks_count, 0)
        self.assertEqual(len(audit_record.retrieved_chunks_summary), audit_record.retrieved_chunks_count)


        # 5. Verify Audit Record integrity
        self.assertEqual(audit_record.request_id, "REQ_INTEGRATION_TEST_001")
        self.assertEqual(audit_record.ml_prediction, decision_result.ml_prediction)
        self.assertEqual(audit_record.approval_probability, decision_result.approval_probability)

        # 6. Verify No Secrets in Audit Record
        audit_dict = audit_record.model_dump()
        for key, val in audit_dict.items():
            val_str = str(val)
            self.assertNotIn("gsk_", val_str)
            self.assertNotIn("secret", val_str.lower())


if __name__ == "__main__":
    unittest.main()
