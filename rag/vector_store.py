"""
Persistent Local Vector Database Layer using ChromaDB.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Manages the local persistent ChromaDB collection for bank policy chunks.
- Stores vector embeddings alongside chunk text and audit metadata.
- Executes similarity search queries and returns ranked policy clauses with provenance.
- Supports deletion and deactivation of documents from the vector index.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from rag.config import RAGConfig, default_rag_config
from rag.embedding_service import BaseEmbeddingService, get_embedding_service


class ChromaVectorStore:
    """Encapsulates persistent ChromaDB operations for policy knowledge retrieval."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
    ):
        self.config = config or default_rag_config
        self.embedding_service = embedding_service or get_embedding_service(self.config)

        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": self.config.DISTANCE_METRIC},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and insert document chunks into ChromaDB collection."""
        if not chunks:
            return 0

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for item in chunks:
            ids.append(item["chunk_id"])
            texts.append(item["text"])
            # Ensure metadata values are JSON-serializable primitives for Chroma
            sanitized_meta = {}
            for k, v in item["metadata"].items():
                if v is None:
                    sanitized_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                else:
                    sanitized_meta[k] = str(v)
            metadatas.append(sanitized_meta)

        # Generate embeddings via embedding service
        embeddings = self.embedding_service.embed_documents(texts)

        # Upsert into Chroma collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector database with natural language and return top-k matching chunks."""
        if not query_text or not query_text.strip():
            return []

        k = top_k or self.config.DEFAULT_TOP_K
        total_in_collection = self.collection.count()
        if total_in_collection == 0:
            return []

        actual_k = min(k, total_in_collection)

        # Compute query embedding
        query_emb = self.embedding_service.embed_query(query_text)

        # Prepare filter (e.g. status == "active")
        where_clause = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=actual_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx] if results["metadatas"] else {}
                distance = results["distances"][0][idx] if results["distances"] else 0.0

                # Convert cosine distance to similarity score (similarity = 1 - distance for cosine)
                similarity_score = round(max(0.0, 1.0 - float(distance)), 4) if self.config.DISTANCE_METRIC == "cosine" else round(float(distance), 4)

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id", "unknown"),
                    "document_name": meta.get("document_name", "unknown"),
                    "document_type": meta.get("document_type", "unknown"),
                    "source_file": meta.get("source_file", "unknown"),
                    "page_number": int(meta.get("page_number", -1)),
                    "policy_version": meta.get("policy_version", "unknown"),
                    "status": meta.get("status", "unknown"),
                    "effective_date": meta.get("effective_date", "unknown"),
                    "relevance_score": similarity_score,
                    "distance": round(float(distance), 4),
                    "text": text,
                    "metadata": meta,
                })

        # Sort descending by relevance score
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return formatted_results

    def update_document_status(self, document_id: str, new_status: str) -> int:
        """Update the status metadata for all chunks of a specific document in ChromaDB."""
        try:
            existing = self.collection.get(where={"document_id": document_id})
            if existing and existing["ids"]:
                updated_metadatas = []
                for meta in existing["metadatas"]:
                    updated_meta = dict(meta)
                    updated_meta["status"] = new_status
                    updated_metadatas.append(updated_meta)
                self.collection.update(ids=existing["ids"], metadatas=updated_metadatas)
                return len(existing["ids"])
            return 0
        except Exception:
            return 0

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID from ChromaDB."""
        try:
            # Query existing IDs for this document
            existing = self.collection.get(where={"document_id": document_id})
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                return len(existing["ids"])
            return 0
        except Exception as e:
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retrieve overview metrics for the Chroma collection."""
        count = self.collection.count()
        return {
            "collection_name": self.config.COLLECTION_NAME,
            "total_chunks_indexed": count,
            "embedding_dimension": self.embedding_service.dimension,
            "distance_metric": self.config.DISTANCE_METRIC,
            "vectorstore_path": str(self.config.VECTORSTORE_DIR),
        }
