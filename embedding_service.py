"""
Embedding Service Module.
Handles text embedding using Nomic-v1.5 model via sentence-transformers.
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


class EmbeddingService:
    """
    Service for generating text embeddings using Nomic-v1.5.

    Uses sentence-transformers to load and run the Nomic embedding model,
    producing 768-dimensional vectors.
    """

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if self._initialized:
            return

        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.model: Optional[SentenceTransformer] = None
        self._initialized = True
        logger.info(f"EmbeddingService initialized with model: {model_name}")

    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model (lazy loading)."""
        if self.model is not None:
            return self.model

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingError(f"Model loading failed: {str(e)}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for multiple documents."""
        if not texts:
            return np.array([])

        batch_size = batch_size or self.batch_size

        try:
            model = self._load_model()
            logger.info(f"Generating embeddings for {len(texts)} documents")

            # Nomic models require 'search_document:' prefix for documents
            prefixed_texts = [f"search_document: {text}" for text in texts]

            embeddings = model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        if not query or not query.strip():
            raise EmbeddingError("Query cannot be empty")

        try:
            model = self._load_model()

            # Nomic models require 'search_query:' prefix for queries
            prefixed_query = f"search_query: {query}"

            embedding = model.encode(
                prefixed_query, convert_to_numpy=True, normalize_embeddings=True
            )

            logger.debug(f"Generated query embedding shape: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise EmbeddingError(f"Query embedding failed: {str(e)}")

    def embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings with appropriate prefix based on type."""
        if is_query:
            return np.array([self.embed_query(t) for t in texts])
        else:
            return self.embed_documents(texts)

    def compute_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "is_loaded": self.model is not None,
        }


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService instance."""
    return EmbeddingService()


def embed_texts(texts: List[str]) -> np.ndarray:
    """Convenience function to embed a list of texts."""
    service = get_embedding_service()
    return service.embed_documents(texts)


def embed_query(query: str) -> np.ndarray:
    """Convenience function to embed a query."""
    service = get_embedding_service()
    return service.embed_query(query)
