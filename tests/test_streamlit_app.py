"""
Unit & Integration Tests for Streamlit Application and UI Modules.

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
from agents.schemas import AuditTrailRecord, DecisionRecommendation, DecisionSupportResult, PolicyStatus
from ui.forms import get_feature_categories, get_sample_applicants, validate_applicant_data


class TestStreamlitAppIntegration(unittest.TestCase):
    """Test suite verifying Streamlit UI modules and end-to-end evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.categories = get_feature_categories()
        cls.orchestrator = LoanEvaluationOrchestrator()
        cls.presets = get_sample_applicants()

    def test_feature_categories_count(self):
        """Verify all 16 features have valid non-empty category lists."""
        self.assertEqual(len(self.categories), 16)
        for feat, options in self.categories.items():
            self.assertGreater(len(options), 0, f"Empty categories for {feat}")

    def test_sample_presets_validation(self):
        """Verify all preset profiles pass strict 16-attribute validation."""
        for name, profile in self.presets.items():
            if not profile:  # Skip header placeholder
                continue
            is_valid, errs = validate_applicant_data(profile, self.categories)
            self.assertTrue(is_valid, f"Preset '{name}' failed validation: {errs}")

    def test_invalid_input_rejection(self):
        """Verify that missing or invalid fields are detected."""
        invalid_profile = {
            "Age Group": "INVALID_AGE",
            # Missing remaining 15 fields
        }
        is_valid, errs = validate_applicant_data(invalid_profile, self.categories)
        self.assertFalse(is_valid)
        self.assertGreater(len(errs), 10)

    def test_end_to_end_evaluation_case1_tp(self):
        """Verify Case 1 evaluation through orchestrator."""
        case_1 = self.presets["Case 1 (True Positive): High Income, Complete Docs, Collateral Available"]
        decision_result, audit_record = self.orchestrator.evaluate_application(case_1, request_id="TEST_CASE_1")

        self.assertIsInstance(decision_result, DecisionSupportResult)
        self.assertIsInstance(audit_record, AuditTrailRecord)
        self.assertEqual(decision_result.ml_prediction, "Approved")
        self.assertGreater(decision_result.approval_probability, 0.50)
        self.assertIsInstance(decision_result.policy_status, PolicyStatus)
        self.assertIsInstance(decision_result.final_recommendation, DecisionRecommendation)


    def test_end_to_end_evaluation_case2_tn(self):
        """Verify Case 2 evaluation through orchestrator."""
        case_2 = self.presets["Case 2 (True Negative): Extreme Loan Request, Incomplete Docs, Contract Job"]
        decision_result, audit_record = self.orchestrator.evaluate_application(case_2, request_id="TEST_CASE_2")

        self.assertIsInstance(decision_result, DecisionSupportResult)
        self.assertEqual(decision_result.ml_prediction, "Rejected")
        self.assertGreater(decision_result.rejection_probability, 0.50)

    def test_no_secret_leakage_in_ui_structures(self):
        """Verify that no API keys or secrets are exposed in rendered structures."""
        sample_case = self.presets["Case 1 (True Positive): High Income, Complete Docs, Collateral Available"]
        decision_result, audit_record = self.orchestrator.evaluate_application(sample_case)

        audit_dump = audit_record.model_dump()
        for key, val in audit_dump.items():
            val_str = str(val)
            self.assertNotIn("gsk_", val_str)
            self.assertNotIn("groq_api_key", val_str.lower())
            self.assertNotIn("secret", val_str.lower())


if __name__ == "__main__":
    unittest.main()
