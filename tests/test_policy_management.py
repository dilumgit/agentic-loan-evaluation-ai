"""
Unit & Integration Tests for Bank Officer Policy Management & RAG Ingestion.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

from pathlib import Path
import sys
import tempfile
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rag.config import default_rag_config
from rag.document_processor import DocumentProcessor
from rag.embedding_service import get_embedding_service
from rag.ingest import DocumentIngestionManager
from rag.retriever import PolicyRetriever
from rag.vector_store import ChromaVectorStore


class TestPolicyManagement(unittest.TestCase):
    """Test suite verifying document validation, deduplication, lifecycle, and retrieval."""

    @classmethod
    def setUpClass(cls):
        cls.config = default_rag_config
        cls.processor = DocumentProcessor(cls.config)
        cls.embedding_service = get_embedding_service(cls.config)
        cls.vector_store = ChromaVectorStore(cls.config, cls.embedding_service)
        cls.ingest_mgr = DocumentIngestionManager(cls.config, cls.processor, cls.vector_store)
        cls.retriever = PolicyRetriever(cls.config, cls.vector_store)

    def test_unsupported_file_rejection(self):
        """Verify that unsupported file extensions (e.g. .exe, .csv, .py) are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "malicious_script.exe"
            bad_file.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")
            is_valid, msg = self.processor.validate_file(bad_file)
            self.assertFalse(is_valid)
            self.assertIn("Unsupported file type", msg)

    def test_empty_document_rejection(self):
        """Verify that 0-byte files are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_file = Path(tmp_dir) / "empty_policy.txt"
            empty_file.touch()
            is_valid, msg = self.processor.validate_file(empty_file)
            self.assertFalse(is_valid)
            self.assertIn("empty", msg.lower())

    def test_sandbox_ingestion_deduplication_and_lifecycle(self):
        """Verify end-to-end sandbox lifecycle: ingestion, deduplication, status update, and deletion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_doc = Path(tmp_dir) / "Sandbox_Secured_Loan_Policy.txt"
            test_doc.write_text(
                "Clause 101: Mandatory Collateral Requirements.\n"
                "All commercial credit applications exceeding Rs. 5,000,000 must be secured by primary tangible immovable property.\n"
                "The maximum permissible loan-to-value ratio for commercial property is 70%.\n\n"
                "Clause 102: Debt-to-Income Limits.\n"
                "Applicants must demonstrate a minimum monthly debt service coverage ratio of 1.5.",
                encoding="utf-8",
            )

            # 1. Ingest Test Document
            ingest_res = self.ingest_mgr.ingest_file(
                file_path=test_doc,
                document_name="Sandbox Secured Loan Policy",
                policy_version="1.0-TEST",
                effective_date="2026-01-01",
                status="active",
                description="Test sandbox policy for automated verification.",
                copy_to_storage=False,
            )

            self.assertTrue(ingest_res["success"], f"Ingestion failed: {ingest_res.get('error')}")
            doc_id = ingest_res["document_id"]
            self.assertGreater(ingest_res["chunks_indexed"], 0)

            # 2. Test Duplicate Hash Detection
            dup_res = self.ingest_mgr.ingest_file(
                file_path=test_doc,
                document_name="Duplicate Attempt",
                copy_to_storage=False,
            )
            self.assertFalse(dup_res["success"])
            self.assertTrue(dup_res["is_duplicate"])
            self.assertEqual(dup_res["document_id"], doc_id)

            # 3. Test Active Policy Retrieval
            active_results = self.retriever.retrieve(
                query="What is the mandatory collateral requirement for commercial loans?",
                top_k=2,
                filter_active_only=True,
            )
            self.assertGreater(len(active_results), 0)
            top_match = active_results[0]
            self.assertIn("immovable property", top_match["text"].lower())
            self.assertEqual(top_match["status"], "active")

            # 4. Test Lifecycle Status Transition to Superseded
            self.ingest_mgr.set_document_status(doc_id, "superseded")
            meta = self.ingest_mgr.get_document_metadata(doc_id)
            self.assertEqual(meta["status"], "superseded")

            # Verify that filter_active_only=True no longer retrieves superseded document
            filtered_results = self.retriever.retrieve(
                query="What is the mandatory collateral requirement for commercial loans?",
                top_k=2,
                filter_active_only=True,
            )
            matching_ids = [r["document_id"] for r in filtered_results if r["document_id"] == doc_id]
            self.assertEqual(len(matching_ids), 0, "Superseded document should not appear in active retrieval!")

            # 5. Clean Sandbox Deletion
            deleted = self.ingest_mgr.delete_document(doc_id, delete_source_file=False)
            self.assertTrue(deleted)
            self.assertIsNone(self.ingest_mgr.get_document_metadata(doc_id))

    def test_production_repository_policy_documents(self):
        """Confirm production repository contains the 5 active institutional underwriting policy documents."""
        prod_docs = [f for f in self.config.DOCUMENTS_DIR.glob("*.pdf")]
        self.assertEqual(len(prod_docs), 5, "Production rag/documents/ must contain the 5 active underwriting policy documents!")
        expected_policy_keywords = ["AGRICULTURAL", "GENERAL", "HOUSING", "PERSONAL", "VEHICLE"]
        doc_names = [f.name.upper() for f in prod_docs]
        for kw in expected_policy_keywords:
            self.assertTrue(
                any(kw in name for name in doc_names),
                f"Missing expected policy document with keyword '{kw}' in rag/documents/",
            )
        # Verify that all documents are non-empty
        for doc_file in prod_docs:
            self.assertGreater(doc_file.stat().st_size, 1000, f"Policy document {doc_file.name} is too small or corrupt.")



if __name__ == "__main__":
    unittest.main()
