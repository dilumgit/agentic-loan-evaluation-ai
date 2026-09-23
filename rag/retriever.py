"""
Policy Retrieval Module.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Accepts natural language credit policy queries from users or future Multi-Agent layers.
- Retrieves ranked policy clauses from ChromaDB with provenance metadata (source document, page, score).
- Formats retrieved evidence for downstream inspection by credit officers and reasoning agents.
- STRICTLY RETRIEVAL ONLY: Does not generate autonomous decisions or execute arbitrary logic.
"""

from typing import Any, Dict, List, Optional

from rag.config import RAGConfig, default_rag_config
from rag.vector_store import ChromaVectorStore


class PolicyRetriever:
    """High-level interface for querying the bank policy knowledge base."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ):
        self.config = config or default_rag_config
        self.vector_store = vector_store or ChromaVectorStore(self.config)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_active_only: bool = True,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant policy clauses matching the natural language query."""
        if not query or not query.strip():
            return []

        filter_meta: Dict[str, Any] = {}
        if filter_active_only:
            filter_meta["status"] = "active"
        if document_id:
            filter_meta["document_id"] = document_id

        results = self.vector_store.query(
            query_text=query,
            top_k=top_k or self.config.DEFAULT_TOP_K,
            filter_metadata=filter_meta if filter_meta else None,
        )

        return results

    def format_retrieved_evidence(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved policy clauses into an evidence text block with provenance citations."""
        if not results:
            return "No matching bank policy clauses found in the knowledge base."

        evidence_blocks = []
        for rank, r in enumerate(results, 1):
            doc_name = r.get("document_name", "Unknown Document")
            page_num = r.get("page_number", -1)
            page_str = f"Page {page_num}" if page_num != -1 else "N/A"
            chunk_id = r.get("chunk_id", "unknown")
            score = r.get("relevance_score", 0.0)
            text = r.get("text", "").strip()

            block = (
                f"[Policy Evidence {rank}]\n"
                f"Document: {doc_name} ({page_str})\n"
                f"Clause ID: {chunk_id} | Relevance Score: {score:.4f}\n"
                f"Content:\n{text}\n"
            )
            evidence_blocks.append(block)

        return "\n" + "-" * 50 + "\n" + "\n".join(evidence_blocks) + "-" * 50 + "\n"
