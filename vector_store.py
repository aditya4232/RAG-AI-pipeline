"""
Vector Store Module.
Handles ChromaDB operations for storing and retrieving document embeddings.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
import numpy as np

from config import (
    CHROMA_PERSIST_PATH,
    CHROMA_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
    SIMILARITY_METRIC,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    page_number: int
    chunk_index: int
    source_type: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class SearchResults:
    """Container for search results."""

    results: List[SearchResult]
    query: str
    total_results: int
    search_time_ms: float


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""

    pass


class VectorStore:
    """
    ChromaDB-based vector store for document embeddings.

    Provides persistent storage of document chunks and their embeddings,
    with support for similarity search and document filtering.
    """

    _instance: Optional["VectorStore"] = None

    def __new__(cls, persist_path: str = CHROMA_PERSIST_PATH):
        """Singleton pattern for vector store."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        persist_path: str = CHROMA_PERSIST_PATH,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        if self._initialized:
            return

        self.persist_path = persist_path
        self.collection_name = collection_name
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection = None
        self._initialized = True

        self._init_client()
        logger.info(f"VectorStore initialized at: {persist_path}")

    def _init_client(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": SIMILARITY_METRIC}
            )

            logger.info(
                f"Collection '{self.collection_name}' ready with {self.collection.count()} documents"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorStoreError(f"ChromaDB initialization failed: {str(e)}")

    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Add document chunks to the vector store."""
        if len(chunk_ids) == 0:
            return 0

        try:
            embeddings_list = embeddings.tolist()

            timestamp = datetime.now(timezone.utc).isoformat()
            for meta in metadatas:
                meta["indexed_at"] = timestamp

            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings_list,
                documents=contents,
                metadatas=metadatas,
            )

            logger.info(f"Added {len(chunk_ids)} chunks to vector store")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise VectorStoreError(f"Document insertion failed: {str(e)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RETRIEVAL,
        document_filter: Optional[str] = None,
    ) -> SearchResults:
        """Search for similar documents."""
        import time

        start_time = time.time()

        try:
            where_clause = None
            if document_filter:
                where_clause = {"document_id": document_filter}

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            search_time_ms = (time.time() - start_time) * 1000

            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    content = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results["distances"] else 0

                    score = (
                        1 - distance
                        if SIMILARITY_METRIC == "cosine"
                        else 1 / (1 + distance)
                    )

                    search_results.append(
                        SearchResult(
                            chunk_id=chunk_id,
                            document_id=metadata.get("document_id", ""),
                            document_name=metadata.get("document_name", ""),
                            content=content,
                            page_number=metadata.get("page_number", 0),
                            chunk_index=metadata.get("chunk_index", 0),
                            source_type=metadata.get("source_type", "text"),
                            score=score,
                            metadata=metadata,
                        )
                    )

            logger.info(
                f"Search completed: {len(search_results)} results in {search_time_ms:.2f}ms"
            )

            return SearchResults(
                results=search_results,
                query="",
                total_results=len(search_results),
                search_time_ms=search_time_ms,
            )

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        try:
            results = self.collection.get(
                where={"document_id": document_id}, include=[]
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(
                    f"Deleted {deleted_count} chunks for document: {document_id}"
                )
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise VectorStoreError(f"Delete failed: {str(e)}")

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all indexed documents."""
        try:
            results = self.collection.get(include=["metadatas"])

            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", ""),
                        "indexed_at": metadata.get("indexed_at", ""),
                        "chunk_count": 0,
                    }
                if doc_id:
                    documents[doc_id]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Failed to get document list: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            documents = self.get_document_list()

            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "total_documents": len(documents),
                "persist_path": self.persist_path,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}

    def reset_collection(self) -> None:
        """Delete and recreate the collection. Warning: This will delete all data!"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": SIMILARITY_METRIC}
            )
            logger.warning("Collection reset - all data deleted")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise VectorStoreError(f"Reset failed: {str(e)}")


def get_vector_store() -> VectorStore:
    """Get the singleton VectorStore instance."""
    return VectorStore()
