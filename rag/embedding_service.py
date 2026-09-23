"""
Embedding Service Interface & Local Sentence-Transformers Provider.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Provides a clean, decoupled embedding interface for document chunks and user queries.
- Defaults to a lightweight, deterministic local model (all-MiniLM-L6-v2) requiring no external API keys.
- Allows seamless future extension to alternative local or remote API-based embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from rag.config import RAGConfig, default_rag_config


class BaseEmbeddingService(ABC):
    """Abstract Base Class for Embedding Providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vector embeddings for a list of document chunk texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute dense vector embedding for a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        pass


class LocalSentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Local SentenceTransformer Embedding Provider (e.g. all-MiniLM-L6-v2)."""

    def __init__(self, model_name: Optional[str] = None, config: Optional[RAGConfig] = None):
        self.config = config or default_rag_config
        self.model_name = model_name or self.config.EMBEDDING_MODEL_NAME
        self._model = None
        self._dim = self.config.EMBEDDING_DIMENSION

    def _lazy_load_model(self):
        """Lazy load the sentence transformer model upon first embedding request."""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. Please install it via 'pip install sentence-transformers'."
                )
            self._model = SentenceTransformer(self.model_name)
            # Auto-detect dimension
            test_emb = self._model.encode("test", convert_to_numpy=True)
            self._dim = int(test_emb.shape[0])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch encode document chunks."""
        if not texts:
            return []
        self._lazy_load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encode a single query string."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        self._lazy_load_model()
        embedding = self._model.encode(
            text.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self._dim


def get_embedding_service(config: Optional[RAGConfig] = None) -> BaseEmbeddingService:
    """Factory function to instantiate the configured embedding service."""
    cfg = config or default_rag_config
    if cfg.EMBEDDING_PROVIDER == "sentence_transformers":
        return LocalSentenceTransformerEmbeddingService(
            model_name=cfg.EMBEDDING_MODEL_NAME, config=cfg
        )
    else:
        # Default fallback
        return LocalSentenceTransformerEmbeddingService(config=cfg)
