"""
LLM Service Module.
Handles interaction with cloud LLMs for answer generation.

Active  : Groq Cloud  (llama-3.3-70b-versatile)
Future  : HuggingFace Inference API (uncomment when ready)
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import httpx

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    # -- HuggingFace (future) --
    # HF_API_TOKEN,
    # HF_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of LLM generation."""
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_ms: float


class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass


SYSTEM_PROMPT = """You are a helpful document assistant. Your task is to answer questions based ONLY on the provided context from documents.

Rules:
1. Answer based ONLY on the information in the provided context
2. If the context doesn't contain enough information, say so clearly
3. Always cite the page numbers where you found the information
4. Be concise and accurate
5. Do not make up information that is not in the context"""

ANSWER_TEMPLATE = """Context from documents:
{context}

---

Question: {question}

Please provide a clear answer based on the context above. Include page number citations."""


class LLMService:
    """
    Service for generating answers using cloud LLMs.

    Supported providers
    -------------------
    groq         - Groq Cloud  (active, free tier, very fast)
    huggingface  - HuggingFace Inference API (commented out for future use)
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "groq":
            self.model = GROQ_MODEL
            self.api_key = GROQ_API_KEY
            self.base_url = "https://api.groq.com/openai/v1"

        # -- HuggingFace (future) -- uncomment below & comment out groq above
        # elif self.provider == "huggingface":
        #     from config import HF_API_TOKEN, HF_MODEL
        #     self.model = HF_MODEL
        #     self.api_key = HF_API_TOKEN
        #     self.base_url = "https://router.huggingface.co/hf-inference"

        else:
            raise LLMError(
                f"Unknown provider: '{self.provider}'. "
                f"Supported: 'groq'. (HuggingFace coming soon.)"
            )

        if not self.api_key:
            logger.warning(
                f"No API key set for '{self.provider}'. "
                f"Set the env variable or update config.py."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"LLMService ready: provider={self.provider}, model={self.model}")

    def _get_chat_url(self) -> str:
        """Build the chat-completions URL for the active provider."""
        # Groq uses the standard OpenAI-compatible path
        return f"{self.base_url}/chat/completions"
        # -- HuggingFace (future) --
        # if self.provider == "huggingface":
        #     return f"{self.base_url}/models/{self.model}/v1/chat/completions"

    def _call_chat_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call the chat completions API."""
        url = self._get_chat_url()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:300]
            raise LLMError(f"API error ({e.response.status_code}): {error_body}")
        except httpx.ConnectError:
            raise LLMError(f"Cannot connect to {self.provider} API at {url}")

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            page = chunk.get("page_number", "N/A")
            doc = chunk.get("document_name", "Unknown")
            stype = chunk.get("source_type", "text")
            parts.append(f"[Source {i} - {doc}, Page {page}, Type: {stype}]\n{content}")
        return "\n\n".join(parts)

    def build_prompt(self, chunks: List[Dict[str, Any]], question: str) -> str:
        """Build the full prompt for the LLM."""
        context = self.build_context(chunks)
        return ANSWER_TEMPLATE.format(context=context, question=question)

    def generate_answer(
        self, chunks: List[Dict[str, Any]], question: str
    ) -> GenerationResult:
        """Generate an answer using the cloud LLM."""
        start = time.time()
        try:
            prompt = self.build_prompt(chunks, question)
            logger.info(f"Generating answer via {self.provider}/{self.model}: {question[:50]}...")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            result = self._call_chat_api(messages)
            gen_time = (time.time() - start) * 1000

            answer = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            logger.info(f"Answer generated in {gen_time:.2f}ms via {self.provider}")

            return GenerationResult(
                answer=answer,
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                generation_time_ms=gen_time,
            )
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            raise LLMError(f"Answer generation failed: {e}")

    def generate_with_retry(
        self, chunks: List[Dict[str, Any]], question: str, max_retries: int = 2
    ) -> GenerationResult:
        """Generate answer with retry logic."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.generate_answer(chunks, question)
            except LLMError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1)
        raise last_error

    def check_health(self) -> Dict[str, Any]:
        """Check if the cloud LLM service is available."""
        if not self.api_key:
            return {
                "status": "unhealthy",
                "error": f"No API key configured for {self.provider}",
                "provider": self.provider,
                "model": self.model,
            }
        try:
            self._call_chat_api([{"role": "user", "content": "Hi"}], max_tokens=5)
            return {
                "status": "healthy",
                "provider": self.provider,
                "model": self.model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": self.provider,
                "model": self.model,
            }


def get_llm_service() -> LLMService:
    """Get an LLMService instance."""
    return LLMService()


def generate_answer(chunks: List[Dict[str, Any]], question: str) -> GenerationResult:
    """Quick helper to generate an answer."""
    return get_llm_service().generate_answer(chunks, question)
