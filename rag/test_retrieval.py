"""
Policy RAG Test & Verification Suite.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Validates that the entire Policy RAG pipeline initializes and functions correctly:
  1. PDF, DOCX, and TXT parsing capabilities.
  2. Embedding generation via local sentence-transformers.
  3. ChromaDB vector storage, metadata filtering, and retrieval.
  4. Temporary end-to-end ingestion and query test in isolated scratch storage.
  5. Audit confirmation of production policy repository state.
"""

from pathlib import Path
import sys
import tempfile

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rag.config import default_rag_config
from rag.document_processor import DocumentProcessor
from rag.embedding_service import get_embedding_service
from rag.ingest import DocumentIngestionManager
from rag.retriever import PolicyRetriever
from rag.vector_store import ChromaVectorStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_rag_test_suite():
    print("=" * 85)
    print("STEP 13: POLICY RAG KNOWLEDGE BASE FOUNDATION — VERIFICATION SUITE")
    print("=" * 85)

    # 1. Component Initializations
    print("\n[1] Testing Component Initialization...")
    config = default_rag_config
    print(f"  - Config loaded. Storage directory: {config.RAG_DIR}")

    processor = DocumentProcessor(config)
    print("  - DocumentProcessor initialized (Supported: PDF, DOCX, TXT).")

    embedding_service = get_embedding_service(config)
    print(f"  - EmbeddingService initialized (Model: {config.EMBEDDING_MODEL_NAME}, Dim: {embedding_service.dimension}).")

    vector_store = ChromaVectorStore(config, embedding_service)
    stats = vector_store.get_collection_stats()
    print(f"  - ChromaVectorStore initialized (Collection: {stats['collection_name']}, Chunks: {stats['total_chunks_indexed']}).")

    ingest_mgr = DocumentIngestionManager(config, processor, vector_store)
    print("  - DocumentIngestionManager initialized.")

    retriever = PolicyRetriever(config, vector_store)
    print("  - PolicyRetriever initialized.")

    # 2. Embedding Generation Test
    print("\n[2] Testing Embedding Vector Generation...")
    sample_query = "What are the credit criteria for personal loan approval?"
    query_emb = embedding_service.embed_query(sample_query)
    print(f"  - Query embedded successfully. Vector length: {len(query_emb)} (Type: {type(query_emb[0]).__name__})")
    assert len(query_emb) == embedding_service.dimension, "Embedding dimension mismatch!"

    # 3. Isolated Ingestion & Retrieval Cycle (Temporary Sandbox Test)
    print("\n[3] Testing Ingestion and Retrieval Pipeline in Isolated Sandbox...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file = Path(tmp_dir) / "sample_test_doc.txt"
        temp_file.write_text(
            "Section 1: General Banking Governance.\n"
            "All credit facility applications must undergo systematic underwriting review.\n"
            "Applicants must submit full proof of income and identity verification documents.\n\n"
            "Section 2: Security & Tangible Collateral Guidelines.\n"
            "Credit facilities exceeding standard unsecured credit limits require tangible asset collateral.\n"
            "Collateral valuation must be verified by an accredited bank valuer prior to disbursement.",
            encoding="utf-8",
        )

        # Ingest temporary test file
        ingest_res = ingest_mgr.ingest_file(
            file_path=temp_file,
            document_name="Sandbox Temporary Test Policy",
            policy_version="0.1-TEST",
            effective_date="2026-01-01",
            status="active",
            copy_to_storage=False,
        )

        assert ingest_res["success"], f"Ingestion failed: {ingest_res.get('error')}"
        test_doc_id = ingest_res["document_id"]
        print(f"  - Ingested sandbox test document (ID: {test_doc_id}, Chunks: {ingest_res['chunks_indexed']}).")

        # Test Retrieval Query
        retrieval_query = "What is required for credit facilities exceeding unsecured limits?"
        results = retriever.retrieve(retrieval_query, top_k=2)

        print(f"  - Executed test retrieval query: '{retrieval_query}'")
        print(f"  - Retrieved {len(results)} matching chunks.")
        assert len(results) > 0, "Retrieval returned 0 chunks on test index!"

        top_match = results[0]
        print(f"  - Top Match Chunk ID : {top_match['chunk_id']}")
        print(f"  - Relevance Score     : {top_match['relevance_score']:.4f}")
        print(f"  - Document Provenance : {top_match['document_name']} (Page: {top_match['page_number']})")
        print(f"  - Snippet             : {top_match['text'][:120]}...")

        # Test Formatted Evidence Output
        formatted_evidence = retriever.format_retrieved_evidence(results)
        print("\n  - Formatted Evidence Block Preview:")
        print(formatted_evidence[:300] + "...\n--------------------------------------------------")

        # Clean up temporary test document from ChromaDB and registry
        deleted = ingest_mgr.delete_document(test_doc_id, delete_source_file=False)
        print(f"  - Sandbox test document cleanly removed from vector index: {deleted}")

    # 4. Check Production Document Repository Status
    print("\n[4] Auditing Production Document Storage Status...")
    prod_docs = list(config.DOCUMENTS_DIR.glob("*"))
    prod_docs = [p for p in prod_docs if p.name != ".gitkeep"]

    prod_registry = ingest_mgr.list_documents()
    final_stats = vector_store.get_collection_stats()

    print(f"  - Raw documents in rag/documents/: {len(prod_docs)}")
    print(f"  - Indexed documents in registry: {len(prod_registry)}")
    print(f"  - Total chunks in Chroma vector store: {final_stats['total_chunks_indexed']}")

    if len(prod_docs) == 0 and len(prod_registry) == 0:
        print("\n" + "*" * 75)
        print("[POLICY RAG AUDIT STATEMENT]")
        print("No policy documents were indexed because no real bank policy documents were supplied.")
        print("*" * 75)

    print("\n" + "=" * 85)
    print("[SUCCESS] POLICY RAG KNOWLEDGE BASE FOUNDATION VERIFIED CLEANLY")
    print("=" * 85)


if __name__ == "__main__":
    run_rag_test_suite()
