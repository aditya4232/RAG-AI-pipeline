"""Quick system test - tests each component individually."""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_components():
    print("\n" + "=" * 50)
    print("RAG System - Component Tests")
    print("=" * 50)

    # Test 1: Config
    print("\n[1/6] Config...")
    try:
        import config
        print(f"  OK - Provider : {config.LLM_PROVIDER}")
        print(f"  OK - Model    : {config.GROQ_MODEL}")
        print(f"  OK - Embedding: {config.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"  FAIL - {e}")
        return

    # Test 2: Document Processor
    print("\n[2/6] Document Processor...")
    try:
        from document_processor import DocumentProcessor
        dp = DocumentProcessor()
        if os.path.exists("test_document.pdf"):
            result = dp.process_document("test_document.pdf")
            print(f"  OK - {len(result.chunks)} chunks from {result.total_pages} pages")
        else:
            print("  SKIP - test_document.pdf not found")
    except Exception as e:
        print(f"  FAIL - {e}")

    # Test 3: Embedding Service
    print("\n[3/6] Embedding Service (downloads model on first run)...")
    try:
        from embedding_service import EmbeddingService
        es = EmbeddingService()
        embeddings = es.embed_documents(["Hello world test"])
        print(f"  OK - {len(embeddings[0])} dimensions")
    except Exception as e:
        print(f"  FAIL - {e}")
        return

    # Test 4: Vector Store
    print("\n[4/6] Vector Store...")
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_collection_stats()
        print(f"  OK - {stats.get('total_chunks', 0)} chunks stored")
    except Exception as e:
        print(f"  FAIL - {e}")

    # Test 5: LLM Service (Groq)
    print("\n[5/6] LLM Service (Groq Cloud)...")
    try:
        from llm_service import LLMService
        llm = LLMService()
        result = llm.generate_answer(
            [
                {
                    "content": "The sky is blue.",
                    "page_number": 1,
                    "document_name": "test.pdf",
                    "source_type": "text",
                }
            ],
            "What color is the sky?",
        )
        print(f"  OK - {result.answer[:80]}...")
    except Exception as e:
        print(f"  FAIL - {e}")

    # Test 6: Reranker
    print("\n[6/6] Reranker (BGE-M3, downloads on first run)...")
    try:
        from reranker import Reranker
        rr = Reranker()
        candidates = [
            {
                "content": "AI is artificial intelligence",
                "chunk_id": "1",
                "document_id": "d1",
                "document_name": "doc.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "source_type": "text",
                "score": 0.9,
            },
            {
                "content": "Dogs are pets",
                "chunk_id": "2",
                "document_id": "d1",
                "document_name": "doc.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "source_type": "text",
                "score": 0.8,
            },
        ]
        results = rr.rerank("What is AI?", candidates)
        print(f"  OK - Reranked {len(results)} documents")
    except Exception as e:
        print(f"  FAIL - {e}")

    print("\n" + "=" * 50)
    print("Tests Complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_components()
