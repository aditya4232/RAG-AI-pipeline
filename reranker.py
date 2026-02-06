"""
Reranker Module.
Handles semantic reranking using BGE-M3 model.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    TOP_K_RERANK,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A reranked search result."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    page_number: int
    chunk_index: int
    source_type: str
    original_score: float
    rerank_score: float
    metadata: dict


class RerankerError(Exception):
    """Raised when reranking fails."""

    pass


class Reranker:
    """
    Semantic reranker using BGE-M3 CrossEncoder.

    Takes initial retrieval results and reranks them based on
    semantic relevance to the query using a cross-encoder model.
    """

    _instance: Optional["Reranker"] = None

    def __new__(cls, model_name: str = RERANKER_MODEL_NAME):
        """Singleton pattern for reranker."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        if self._initialized:
            return

        self.model_name = model_name
        self.top_k = TOP_K_RERANK
        self.model: Optional[CrossEncoder] = None
        self._initialized = True
        logger.info(f"Reranker initialized with model: {model_name}")

    def _load_model(self) -> CrossEncoder:
        """Load the reranker model (lazy loading)."""
        if self.model is not None:
            return self.model

        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name, max_length=512, trust_remote_code=True
            )
            logger.info("Reranker model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            raise RerankerError(f"Model loading failed: {str(e)}")

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute relevance scores for query-document pairs."""
        if not documents:
            return []

        try:
            model = self._load_model()
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)

            if isinstance(scores, np.ndarray):
                scores = scores.tolist()

            return scores

        except Exception as e:
            logger.error(f"Score computation failed: {str(e)}")
            raise RerankerError(f"Score computation failed: {str(e)}")

    def rerank(
        self, query: str, candidates: List[dict], top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """Rerank candidate results based on semantic relevance."""
        if not candidates:
            return []

        top_k = top_k or self.top_k

        try:
            documents = [c.get("content", "") for c in candidates]
            scores = self.compute_scores(query, documents)

            ranked_results = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                ranked_results.append(
                    RankedResult(
                        chunk_id=candidate.get("chunk_id", ""),
                        document_id=candidate.get("document_id", ""),
                        document_name=candidate.get("document_name", ""),
                        content=candidate.get("content", ""),
                        page_number=candidate.get("page_number", 0),
                        chunk_index=candidate.get("chunk_index", 0),
                        source_type=candidate.get("source_type", "text"),
                        original_score=candidate.get("score", 0.0),
                        rerank_score=float(score),
                        metadata=candidate.get("metadata", {}),
                    )
                )

            ranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            top_results = ranked_results[:top_k]

            logger.info(
                f"Reranked {len(candidates)} candidates, returning top {len(top_results)}"
            )
            return top_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise RerankerError(f"Reranking failed: {str(e)}")

    def rerank_search_results(
        self, query: str, search_results, top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """Rerank SearchResults from vector store."""
        candidates = []
        for result in search_results.results:
            candidates.append(
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "document_name": result.document_name,
                    "content": result.content,
                    "page_number": result.page_number,
                    "chunk_index": result.chunk_index,
                    "source_type": result.source_type,
                    "score": result.score,
                    "metadata": result.metadata,
                }
            )

        return self.rerank(query, candidates, top_k)


def get_reranker() -> Reranker:
    """Get the singleton Reranker instance."""
    return Reranker()


def rerank_results(
    query: str, candidates: List[dict], top_k: int = TOP_K_RERANK
) -> List[RankedResult]:
    """Convenience function to rerank results."""
    reranker = get_reranker()
    return reranker.rerank(query, candidates, top_k)
