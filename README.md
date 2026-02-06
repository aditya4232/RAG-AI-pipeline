# RAG Document Q&A System

A backend system that lets you upload PDFs and ask questions about them. It finds relevant sections and generates answers with page citations.

## What This Does

1. **Upload a PDF** — The system extracts text and tables, breaks them into chunks, and stores them in a searchable database
2. **Ask a Question** — It finds the most relevant chunks, re-ranks them for accuracy, generates an answer using an LLM, and verifies quality

## Quick Start

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run the server
python main.py
```

Open http://localhost:8000/docs to see the interactive API.

## API Endpoints

| Endpoint | What it does |
|----------|--------------|
| `POST /documents/upload` | Upload a PDF file |
| `GET /documents` | List all uploaded documents |
| `DELETE /documents/{id}` | Remove a document |
| `POST /query` | Ask a question about your documents |
| `GET /health` | Check if the system is running |

## Example Usage

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/documents/upload" -F "file=@your_file.pdf"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## How It Works

The system uses two pipelines:

**Ingestion (when you upload):**
- PyMuPDF extracts text from each page
- Camelot extracts tables
- Text gets split into ~500 word chunks with 50 word overlap
- Nomic-v1.5 creates embeddings (vector representations)
- ChromaDB stores everything persistently

**Query (when you ask):**
- Your question gets embedded the same way
- ChromaDB finds the 10 most similar chunks
- BGE-M3 reranks to get the top 3 most relevant
- **Groq Cloud (Llama 3.3 70B)** generates an answer with citations
- LangGraph verifies the answer quality (factuality, completeness)

## LLM Provider

| Provider | Status | Model | How to get key |
|----------|--------|-------|----------------|
| **Groq** | ✅ Active | `llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) |
| **HuggingFace** | 💬 Commented (future) | `zephyr-7b-beta` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

To switch providers, edit `.env` or `config.py`.

## Configuration

Edit `config.py` to change:
- `LLM_PROVIDER` — Which LLM provider to use (`groq`)
- `GROQ_MODEL` — Which Groq model to use
- `CHUNK_SIZE_WORDS` — How big each chunk is (default: 500)
- `TOP_K_RETRIEVAL` — How many chunks to retrieve (default: 10)
- `TOP_K_RERANK` — How many to keep after reranking (default: 3)

## Models Used

- **Embeddings:** nomic-ai/nomic-embed-text-v1.5 (768 dimensions)
- **Reranker:** BAAI/bge-m3 (cross-encoder for semantic matching)
- **LLM:** Llama 3.3 70B via Groq Cloud (free tier)

## Project Structure

```
config.py              — Settings and constants
document_processor.py  — PDF extraction and chunking
embedding_service.py   — Text to vector conversion
vector_store.py        — ChromaDB storage and search
reranker.py            — Semantic reranking
llm_service.py         — Answer generation via Groq Cloud
workflow.py            — LangGraph quality verification
main.py                — FastAPI server and orchestration
test_system.py         — Component tests
start_server.bat       — One-click server start (Windows)
.env                   — API keys (not committed to git)
```

## Requirements

- Python 3.10+
- Groq API key (free — sign up at https://console.groq.com)
- ~2GB disk space for embedding/reranker model caches

## Troubleshooting

**Slow first run?** The embedding and reranker models download on first use (~2GB total). Subsequent runs are fast.

**Port 8000 busy?** Change the port in main.py: `uvicorn.run(app, host="0.0.0.0", port=YOUR_PORT)`

**API key issues?** Make sure your Groq key is set in `.env` or `config.py`.

## Testing

```bash
python test_system.py
```

This runs through each component and confirms they're working.
