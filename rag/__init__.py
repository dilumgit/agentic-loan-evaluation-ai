"""
Policy RAG Knowledge Base Package.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

from rag.config import RAGConfig, default_rag_config
from rag.document_processor import DocumentProcessor
from rag.embedding_service import BaseEmbeddingService, LocalSentenceTransformerEmbeddingService, get_embedding_service
from rag.ingest import DocumentIngestionManager
from rag.retriever import PolicyRetriever
from rag.vector_store import ChromaVectorStore

__all__ = [
    "RAGConfig",
    "default_rag_config",
    "DocumentProcessor",
    "BaseEmbeddingService",
    "LocalSentenceTransformerEmbeddingService",
    "get_embedding_service",
    "DocumentIngestionManager",
    "ChromaVectorStore",
    "PolicyRetriever",
]
