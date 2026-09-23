"""
Policy RAG Configuration Module.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Centralizes all paths, chunking parameters, embedding settings, and security constraints
  for the Policy Retrieval-Augmented Generation (RAG) subsystem.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass
class RAGConfig:
    """Configuration settings for Policy RAG Knowledge Base."""

    # Workspace Paths
    WORKSPACE_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    RAG_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    DOCUMENTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "documents")
    PROCESSED_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "processed")
    VECTORSTORE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "vectorstore")
    METADATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "metadata")
    REGISTRY_FILE: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "metadata" / "documents_registry.json")

    # Document Processing & Security Constraints
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {".pdf", ".docx", ".txt"})
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB max per policy document

    # Chunking Configuration (Deterministic & Configurable)
    # Default: 500 characters with 50 characters overlap (~100 tokens / 10 tokens overlap)
    # Rationale: Policy retrieval requires fine-grained clause-level granularity rather than monolithic pages.
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_LENGTH: int = 40  # discard tiny whitespace artifacts

    # Embedding Service Configuration
    EMBEDDING_PROVIDER: str = "sentence_transformers"  # 'sentence_transformers' or 'chroma_default'
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector Database (ChromaDB) Configuration
    COLLECTION_NAME: str = "bank_policies"
    DISTANCE_METRIC: str = "cosine"  # 'cosine', 'l2', or 'ip'

    # Retrieval Configuration
    DEFAULT_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.0  # minimum similarity score to return (0.0 = return top_k)

    def __post_init__(self):
        """Ensure all required directories exist."""
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        self.METADATA_DIR.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
default_rag_config = RAGConfig()
