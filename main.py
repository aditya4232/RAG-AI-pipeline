"""
Main Pipeline Module.
Orchestrates document ingestion and question answering pipelines.
Provides FastAPI endpoints for the RAG system.
"""

import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    UPLOAD_DIR,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    LOG_LEVEL,
    LOG_FORMAT,
)
from document_processor import DocumentProcessor, Chunk
from embedding_service import EmbeddingService
from vector_store import VectorStore
from reranker import Reranker
from llm_service import LLMService
from workflow import QualityWorkflow, VerifiedAnswer, ConfidenceScores, SourceCitation

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# API Request/Response Models


class QueryRequest(BaseModel):
    """Request model for question answering."""

    question: str = Field(..., min_length=1, description="The question to answer")
    document_id: Optional[str] = Field(None, description="Filter by specific document")


class SourceCitationResponse(BaseModel):
    """Source citation in response."""

    document_name: str
    page_number: int
    chunk_id: str
    relevance_score: float


class ConfidenceScoresResponse(BaseModel):
    """Confidence scores in response."""

    factuality: float
    citations: float
    completeness: float
    overall: float


class AnswerResponse(BaseModel):
    """Response model for question answering."""

    answer: str
    sources: List[SourceCitationResponse]
    confidence: ConfidenceScoresResponse
    query_time_ms: float
    verification_notes: List[str]


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    document_id: str
    document_name: str
    chunk_count: int
    indexed_at: str


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""

    document_id: str
    document_name: str
    total_pages: int
    chunk_count: int
    processing_time_seconds: float
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    llm_status: str
    vector_store_status: str
    indexed_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Response for document deletion."""

    document_id: str
    chunks_deleted: int
    status: str


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Coordinates document ingestion and question answering flows,
    integrating all system components.
    """

    def __init__(self):
        logger.info("Initializing RAG Pipeline...")

        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.llm_service = LLMService()
        self.quality_workflow = QualityWorkflow()

        self.processing_queue: List[str] = []
        self.processing_status: Dict[str, str] = {}

        logger.info("RAG Pipeline initialized successfully")

    def ingest_document(self, file_path: str) -> IngestionResponse:
        """
        Ingest a document into the RAG system.

        Steps:
        1. Process PDF (extract text & tables)
        2. Chunk content
        3. Generate embeddings
        4. Store in vector database
        """
        start_time = time.time()
        document_name = Path(file_path).name

        try:
            logger.info(f"Starting ingestion: {document_name}")

            doc_result = self.document_processor.process_document(file_path)

            if doc_result.status == "failed":
                return IngestionResponse(
                    document_id=doc_result.document_id,
                    document_name=document_name,
                    total_pages=0,
                    chunk_count=0,
                    processing_time_seconds=time.time() - start_time,
                    status="failed",
                    message=doc_result.error_message or "Processing failed",
                )

            if not doc_result.chunks:
                return IngestionResponse(
                    document_id=doc_result.document_id,
                    document_name=document_name,
                    total_pages=doc_result.total_pages,
                    chunk_count=0,
                    processing_time_seconds=time.time() - start_time,
                    status="failed",
                    message="No content extracted from document",
                )

            chunk_texts = [chunk.content for chunk in doc_result.chunks]
            embeddings = self.embedding_service.embed_documents(chunk_texts)

            chunk_ids = [chunk.chunk_id for chunk in doc_result.chunks]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                    "word_count": chunk.word_count,
                    "timestamp": chunk.timestamp,
                }
                for chunk in doc_result.chunks
            ]

            self.vector_store.add_documents(
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                contents=chunk_texts,
                metadatas=metadatas,
            )

            processing_time = time.time() - start_time
            pages_per_second = (
                doc_result.total_pages / processing_time if processing_time > 0 else 0
            )
            logger.info(
                f"Ingestion complete: {doc_result.total_pages} pages, "
                f"{len(doc_result.chunks)} chunks in {processing_time:.2f}s "
                f"({pages_per_second:.1f} pages/sec)"
            )

            return IngestionResponse(
                document_id=doc_result.document_id,
                document_name=document_name,
                total_pages=doc_result.total_pages,
                chunk_count=len(doc_result.chunks),
                processing_time_seconds=processing_time,
                status="success",
                message=f"Document indexed successfully with {len(doc_result.chunks)} chunks",
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            return IngestionResponse(
                document_id=str(uuid.uuid4()),
                document_name=document_name,
                total_pages=0,
                chunk_count=0,
                processing_time_seconds=time.time() - start_time,
                status="failed",
                message=str(e),
            )

    def add_to_queue(self, file_path: str) -> str:
        """Add a document to the processing queue."""
        queue_id = str(uuid.uuid4())
        self.processing_queue.append(file_path)
        self.processing_status[queue_id] = "queued"
        logger.info(
            f"Document queued: {file_path} (position: {len(self.processing_queue)})"
        )
        return queue_id

    def process_queue(self) -> List[IngestionResponse]:
        """Process all documents in the queue."""
        results = []
        while self.processing_queue:
            file_path = self.processing_queue.pop(0)
            result = self.ingest_document(file_path)
            results.append(result)
        return results

    def answer_question(
        self, question: str, document_filter: Optional[str] = None
    ) -> AnswerResponse:
        """
        Answer a question using the RAG pipeline.

        Steps:
        1. Preprocess query
        2. Generate query embedding
        3. Search vector store (top 10)
        4. Rerank results (top 3)
        5. Generate answer with LLM
        6. Verify answer quality
        7. Format response
        """
        start_time = time.time()

        try:
            logger.info(f"Processing question: {question[:50]}...")

            cleaned_question = self._preprocess_query(question)
            query_embedding = self.embedding_service.embed_query(cleaned_question)

            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=TOP_K_RETRIEVAL,
                document_filter=document_filter,
            )

            if not search_results.results:
                return self._empty_answer_response(
                    "No relevant documents found. Please upload documents first.",
                    start_time,
                )

            ranked_results = self.reranker.rerank_search_results(
                query=cleaned_question,
                search_results=search_results,
                top_k=TOP_K_RERANK,
            )

            if not ranked_results:
                return self._empty_answer_response(
                    "No relevant content found for your question.", start_time
                )

            chunks_for_llm = [
                {
                    "content": r.content,
                    "page_number": r.page_number,
                    "document_name": r.document_name,
                    "source_type": r.source_type,
                    "chunk_id": r.chunk_id,
                    "rerank_score": r.rerank_score,
                }
                for r in ranked_results
            ]

            generation_result = self.llm_service.generate_answer(
                chunks=chunks_for_llm, question=cleaned_question
            )

            verified_answer = self.quality_workflow.verify_answer(
                answer=generation_result.answer,
                question=cleaned_question,
                chunks=chunks_for_llm,
            )

            query_time_ms = (time.time() - start_time) * 1000

            sources = [
                SourceCitationResponse(
                    document_name=s.document_name,
                    page_number=s.page_number,
                    chunk_id=s.chunk_id,
                    relevance_score=s.relevance_score,
                )
                for s in verified_answer.sources
            ]

            confidence = ConfidenceScoresResponse(
                factuality=verified_answer.confidence.factuality,
                citations=verified_answer.confidence.citations,
                completeness=verified_answer.confidence.completeness,
                overall=verified_answer.confidence.overall,
            )

            logger.info(
                f"Question answered in {query_time_ms:.2f}ms, "
                f"confidence: {confidence.overall:.2f}"
            )

            return AnswerResponse(
                answer=verified_answer.answer,
                sources=sources,
                confidence=confidence,
                query_time_ms=query_time_ms,
                verification_notes=verified_answer.verification_notes,
            )

        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _preprocess_query(self, query: str) -> str:
        """Clean and preprocess query string."""
        query = query.strip()
        query = " ".join(query.split())
        return query

    def _empty_answer_response(self, message: str, start_time: float) -> AnswerResponse:
        """Create an empty answer response."""
        return AnswerResponse(
            answer=message,
            sources=[],
            confidence=ConfidenceScoresResponse(
                factuality=0.0, citations=0.0, completeness=0.0, overall=0.0
            ),
            query_time_ms=(time.time() - start_time) * 1000,
            verification_notes=["No documents to search"],
        )

    def get_document_list(self) -> List[DocumentInfo]:
        """Get list of all indexed documents."""
        docs = self.vector_store.get_document_list()
        return [
            DocumentInfo(
                document_id=d["document_id"],
                document_name=d["document_name"],
                chunk_count=d["chunk_count"],
                indexed_at=d["indexed_at"],
            )
            for d in docs
        ]

    def delete_document(self, document_id: str) -> DeleteResponse:
        """Delete a document from the index."""
        deleted_count = self.vector_store.delete_document(document_id)
        return DeleteResponse(
            document_id=document_id,
            chunks_deleted=deleted_count,
            status="success" if deleted_count > 0 else "not_found",
        )

    def get_health(self) -> HealthResponse:
        """Get system health status."""
        llm_health = self.llm_service.check_health()
        vs_stats = self.vector_store.get_collection_stats()

        return HealthResponse(
            status="healthy" if llm_health["status"] == "healthy" else "degraded",
            llm_status=llm_health["status"],
            vector_store_status="healthy" if vs_stats else "unhealthy",
            indexed_documents=vs_stats.get("total_documents", 0),
            total_chunks=vs_stats.get("total_chunks", 0),
        )


# FastAPI Application

pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup, cleanup on shutdown."""
    global pipeline
    pipeline = RAGPipeline()
    logger.info("RAG Pipeline started")
    yield
    logger.info("RAG Pipeline shutting down")


app = FastAPI(
    title="RAG Document Intelligence API",
    description="Document processing and question answering system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    return get_pipeline().get_health()


@app.post("/documents/upload", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = get_pipeline().ingest_document(file_path)
        return result

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    return get_pipeline().get_document_list()


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document from the index."""
    return get_pipeline().delete_document(document_id)


@app.post("/query", response_model=AnswerResponse)
async def answer_question(request: QueryRequest):
    """Answer a question using the indexed documents."""
    return get_pipeline().answer_question(
        question=request.question, document_filter=request.document_id
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 60, flush=True)
    print("RAG Document Intelligence System", flush=True)
    print("=" * 60, flush=True)

    print("\nStarting API server...", flush=True)
    print("(First startup may take 1-2 minutes while loading ML models)", flush=True)
    print("API docs available at: http://localhost:8000/docs", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
