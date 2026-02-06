"""
Document Processor Module.
Handles PDF extraction using PyMuPDF (text) and Camelot (tables),
and text chunking with overlap.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import pandas as pd

from config import (
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents extracted content from a single page."""

    page_number: int
    text: str
    tables: List[pd.DataFrame] = field(default_factory=list)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    page_number: int
    chunk_index: int
    source_type: str  # 'text' or 'table'
    word_count: int
    timestamp: str


@dataclass
class DocumentResult:
    """Result of document processing."""

    document_id: str
    document_name: str
    total_pages: int
    chunks: List[Chunk]
    processing_time_seconds: float
    status: str
    error_message: Optional[str] = None


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""

    pass


class ValidationError(Exception):
    """Raised when file validation fails."""

    pass


class DocumentProcessor:
    """
    Processes PDF documents: extraction, table detection, and chunking.

    Attributes:
        chunk_size: Number of words per chunk (default: 500)
        chunk_overlap: Number of overlapping words between chunks (default: 50)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_WORDS,
        chunk_overlap: int = CHUNK_OVERLAP_WORDS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def validate_file(self, file_path: str) -> None:
        """Validate the uploaded file."""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(f"File not found: {file_path}")

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file type: {path.suffix}. Allowed: {ALLOWED_EXTENSIONS}"
            )

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. Max: {MAX_FILE_SIZE_MB}MB"
            )

        logger.info(f"File validated: {path.name} ({file_size_mb:.2f}MB)")

    def extract_text_with_fitz(self, pdf_path: str) -> List[PageContent]:
        """Extract text from PDF using PyMuPDF (fitz)."""
        pages: List[PageContent] = []

        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Opened PDF with {len(doc)} pages")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                text = self._clean_text(text)

                pages.append(
                    PageContent(page_number=page_num + 1, text=text, tables=[])
                )

            doc.close()
            logger.info(f"Extracted text from {len(pages)} pages")
            return pages

        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            raise PDFExtractionError(f"Text extraction failed: {str(e)}")

    def extract_tables_with_camelot(
        self, pdf_path: str
    ) -> Dict[int, List[pd.DataFrame]]:
        """Extract tables from PDF using Camelot."""
        tables_by_page: Dict[int, List[pd.DataFrame]] = {}

        try:
            import camelot

            tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

            if len(tables) == 0:
                tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

            for table in tables:
                page_num = table.page
                if page_num not in tables_by_page:
                    tables_by_page[page_num] = []
                tables_by_page[page_num].append(table.df)

            total_tables = sum(len(t) for t in tables_by_page.values())
            logger.info(
                f"Extracted {total_tables} tables from {len(tables_by_page)} pages"
            )
            return tables_by_page

        except ImportError:
            logger.warning("Camelot not available, skipping table extraction")
            return {}
        except Exception as e:
            logger.warning(
                f"Table extraction failed (continuing without tables): {str(e)}"
            )
            return {}

    def merge_content(
        self, pages: List[PageContent], tables_by_page: Dict[int, List[pd.DataFrame]]
    ) -> List[PageContent]:
        """Merge extracted text and tables into unified page content."""
        for page in pages:
            if page.page_number in tables_by_page:
                page.tables = tables_by_page[page.page_number]
        return pages

    def chunk_content(
        self, pages: List[PageContent], document_id: str, document_name: str
    ) -> List[Chunk]:
        """Split content into chunks with overlap."""
        chunks: List[Chunk] = []
        chunk_index = 0
        timestamp = datetime.now(timezone.utc).isoformat()

        for page in pages:
            if page.text.strip():
                text_chunks = self._split_into_chunks(page.text)
                for chunk_text in text_chunks:
                    word_count = len(chunk_text.split())
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            document_name=document_name,
                            content=chunk_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            source_type="text",
                            word_count=word_count,
                            timestamp=timestamp,
                        )
                    )
                    chunk_index += 1

            for table_idx, table_df in enumerate(page.tables):
                table_text = self._table_to_text(table_df)
                if table_text.strip():
                    word_count = len(table_text.split())
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            document_name=document_name,
                            content=table_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            source_type="table",
                            word_count=word_count,
                            timestamp=timestamp,
                        )
                    )
                    chunk_index += 1

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def process_document(self, file_path: str) -> DocumentResult:
        """Process a PDF document through the full pipeline."""
        import time

        start_time = time.time()

        document_id = str(uuid.uuid4())
        document_name = Path(file_path).name

        try:
            self.validate_file(file_path)
            pages = self.extract_text_with_fitz(file_path)
            tables_by_page = self.extract_tables_with_camelot(file_path)
            pages = self.merge_content(pages, tables_by_page)
            chunks = self.chunk_content(pages, document_id, document_name)

            processing_time = time.time() - start_time
            logger.info(
                f"Document processed in {processing_time:.2f}s: {len(chunks)} chunks"
            )

            return DocumentResult(
                document_id=document_id,
                document_name=document_name,
                total_pages=len(pages),
                chunks=chunks,
                processing_time_seconds=processing_time,
                status="success",
            )

        except (ValidationError, PDFExtractionError) as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed: {str(e)}")
            return DocumentResult(
                document_id=document_id,
                document_name=document_name,
                total_pages=0,
                chunks=[],
                processing_time_seconds=processing_time,
                status="failed",
                error_message=str(e),
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        lines = text.split("\n")
        cleaned_lines = [" ".join(line.split()) for line in lines]
        text = "\n".join(line for line in cleaned_lines if line)
        return text

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with word-based overlap."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            start = end - self.chunk_overlap
            if start >= len(words):
                break

        return chunks

    def _table_to_text(self, df: pd.DataFrame) -> str:
        """Convert a DataFrame table to readable text."""
        try:
            lines = []
            headers = [str(col) for col in df.columns]
            lines.append(" | ".join(headers))
            lines.append("-" * len(lines[0]))

            for _, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                lines.append(row_text)

            return "\n".join(lines)
        except Exception:
            return df.to_string()


def process_pdf(file_path: str) -> DocumentResult:
    """Convenience function to process a PDF file."""
    processor = DocumentProcessor()
    return processor.process_document(file_path)
