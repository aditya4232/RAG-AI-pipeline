"""
Configuration settings for RAG Document Intelligence System.

Supported LLM Providers:
  - groq        : Free & fast (active) - https://console.groq.com/keys
  - huggingface : Free HF Inference API (future) - https://huggingface.co/settings/tokens
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# -- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
CHROMA_PERSIST_PATH = str(BASE_DIR / "chroma_db")
UPLOAD_DIR = str(BASE_DIR / "uploads")
LOGS_DIR = str(BASE_DIR / "logs")

os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# -- Document Processing ---------------------------------------------------
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".pdf"}
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# -- Embedding Model -------------------------------------------------------
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32

# -- Vector Database -------------------------------------------------------
CHROMA_COLLECTION_NAME = "documents"
TOP_K_RETRIEVAL = 10
SIMILARITY_METRIC = "cosine"

# -- Reranker --------------------------------------------------------------
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TOP_K_RERANK = 3

# -- LLM Provider ----------------------------------------------------------
# Active provider: "groq"
# Future provider: "huggingface" (uncomment when ready)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# --- Groq Cloud (active - free tier, very fast) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- HuggingFace Inference API (future - uncomment & add token) -----------
# HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
# HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# -- LLM Generation Settings -----------------------------------------------
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# -- Quality Verification Thresholds ---------------------------------------
CONFIDENCE_THRESHOLD_LOW = 0.5
CONFIDENCE_THRESHOLD_MEDIUM = 0.7
CONFIDENCE_THRESHOLD_HIGH = 0.85

# -- Performance Targets ---------------------------------------------------
TARGET_INGESTION_TIME_PER_100_PAGES = 30  # seconds
TARGET_QUERY_TIME = 6  # seconds

# -- Logging ---------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
