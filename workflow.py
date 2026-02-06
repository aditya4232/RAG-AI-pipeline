"""
Workflow Module.
LangGraph-based orchestration for quality verification of generated answers.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypedDict
import re

from langgraph.graph import StateGraph, END

from config import (
    CONFIDENCE_THRESHOLD_LOW,
    CONFIDENCE_THRESHOLD_MEDIUM,
    CONFIDENCE_THRESHOLD_HIGH,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class QualityState(TypedDict):
    """State for the quality verification workflow."""

    answer: str
    question: str
    chunks: List[Dict[str, Any]]
    factuality_score: float
    citation_score: float
    completeness_score: float
    overall_confidence: float
    verification_notes: List[str]
    is_verified: bool


@dataclass
class ConfidenceScores:
    """Container for confidence scores."""

    factuality: float
    citations: float
    completeness: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "factuality": self.factuality,
            "citations": self.citations,
            "completeness": self.completeness,
            "overall": self.overall,
        }


@dataclass
class SourceCitation:
    """A source citation from the answer."""

    document_name: str
    page_number: int
    chunk_id: str
    relevance_score: float


@dataclass
class VerifiedAnswer:
    """Final verified answer with quality scores."""

    answer: str
    sources: List[SourceCitation]
    confidence: ConfidenceScores
    verification_notes: List[str]
    is_verified: bool


def check_factuality(state: QualityState) -> QualityState:
    """Check if the answer aligns with the provided chunks."""
    answer = state["answer"].lower()
    chunks = state["chunks"]
    notes = state.get("verification_notes", [])

    if not chunks:
        state["factuality_score"] = 0.0
        notes.append("No source chunks provided")
        state["verification_notes"] = notes
        return state

    all_content = " ".join([c.get("content", "").lower() for c in chunks])
    answer_words = set(re.findall(r"\b\w{4,}\b", answer))
    content_words = set(re.findall(r"\b\w{4,}\b", all_content))

    if not answer_words:
        state["factuality_score"] = 0.5
        notes.append("Answer contains no significant terms to verify")
        state["verification_notes"] = notes
        return state

    overlap = answer_words.intersection(content_words)
    overlap_ratio = len(overlap) / len(answer_words)

    hallucination_phrases = [
        "i think",
        "i believe",
        "probably",
        "might be",
        "i'm not sure",
        "as an ai",
        "i don't have",
    ]
    has_hallucination_indicator = any(
        phrase in answer for phrase in hallucination_phrases
    )

    if has_hallucination_indicator:
        overlap_ratio *= 0.8
        notes.append("Answer contains uncertainty indicators")

    admission_phrases = [
        "not mentioned in",
        "not found in",
        "context doesn't",
        "no information about",
        "cannot find",
    ]
    admits_limitation = any(phrase in answer for phrase in admission_phrases)

    if admits_limitation:
        overlap_ratio = max(overlap_ratio, 0.7)
        notes.append("Answer appropriately acknowledges limitations")

    state["factuality_score"] = min(overlap_ratio, 1.0)
    state["verification_notes"] = notes

    logger.debug(f"Factuality score: {state['factuality_score']:.2f}")
    return state


def check_citations(state: QualityState) -> QualityState:
    """Validate that page references in the answer are accurate."""
    answer = state["answer"]
    chunks = state["chunks"]
    notes = state.get("verification_notes", [])

    page_pattern = r"page\s*(\d+)|p\.\s*(\d+)|pg\.\s*(\d+)"
    cited_pages = re.findall(page_pattern, answer.lower())
    cited_page_nums = set()
    for match in cited_pages:
        for group in match:
            if group:
                cited_page_nums.add(int(group))

    if not cited_page_nums:
        state["citation_score"] = 0.5
        notes.append("No page citations found in answer")
        state["verification_notes"] = notes
        return state

    valid_pages = set(c.get("page_number", 0) for c in chunks)
    valid_citations = cited_page_nums.intersection(valid_pages)

    if cited_page_nums:
        citation_accuracy = len(valid_citations) / len(cited_page_nums)
    else:
        citation_accuracy = 0.0

    if len(valid_citations) < len(cited_page_nums):
        invalid = cited_page_nums - valid_pages
        notes.append(f"Invalid page citations: {invalid}")

    if citation_accuracy == 1.0:
        notes.append("All page citations verified")

    state["citation_score"] = citation_accuracy
    state["verification_notes"] = notes

    logger.debug(f"Citation score: {state['citation_score']:.2f}")
    return state


def check_completeness(state: QualityState) -> QualityState:
    """Check if the answer addresses the question."""
    answer = state["answer"].lower()
    question = state["question"].lower()
    notes = state.get("verification_notes", [])

    question_types = {
        "what": ["what is", "what are", "what was", "what were"],
        "who": ["who is", "who are", "who was", "who were"],
        "when": ["when is", "when was", "when did", "when will"],
        "where": ["where is", "where are", "where was", "where were"],
        "why": ["why is", "why are", "why did", "why was"],
        "how": ["how to", "how does", "how did", "how is", "how many", "how much"],
        "which": ["which is", "which are", "which was", "which were"],
    }

    detected_type = None
    for qtype, patterns in question_types.items():
        if any(pattern in question for pattern in patterns):
            detected_type = qtype
            break

    answer_words = len(answer.split())
    if answer_words < 10:
        length_penalty = 0.7
        notes.append("Answer is very brief")
    elif answer_words < 20:
        length_penalty = 0.9
    else:
        length_penalty = 1.0

    question_words = set(re.findall(r"\b\w{4,}\b", question))
    common_words = {"what", "where", "when", "which", "that", "this", "these", "those"}
    question_words -= common_words

    answer_words_set = set(re.findall(r"\b\w{4,}\b", answer))

    if question_words:
        term_coverage = len(question_words.intersection(answer_words_set)) / len(
            question_words
        )
    else:
        term_coverage = 0.5

    completeness = term_coverage * length_penalty

    structure_indicators = [
        "first",
        "second",
        "additionally",
        "however",
        "therefore",
        "1.",
        "2.",
        "-",
    ]
    has_structure = any(indicator in answer for indicator in structure_indicators)
    if has_structure:
        completeness = min(completeness * 1.1, 1.0)
        notes.append("Answer has good structure")

    state["completeness_score"] = completeness
    state["verification_notes"] = notes

    logger.debug(f"Completeness score: {state['completeness_score']:.2f}")
    return state


def calculate_overall(state: QualityState) -> QualityState:
    """Calculate overall confidence score from individual scores."""
    weights = {"factuality": 0.4, "citations": 0.3, "completeness": 0.3}

    factuality = state.get("factuality_score", 0.0)
    citations = state.get("citation_score", 0.0)
    completeness = state.get("completeness_score", 0.0)

    overall = (
        factuality * weights["factuality"]
        + citations * weights["citations"]
        + completeness * weights["completeness"]
    )

    state["overall_confidence"] = overall
    state["is_verified"] = overall >= CONFIDENCE_THRESHOLD_MEDIUM

    notes = state.get("verification_notes", [])
    if overall >= CONFIDENCE_THRESHOLD_HIGH:
        notes.append("High confidence answer")
    elif overall >= CONFIDENCE_THRESHOLD_MEDIUM:
        notes.append("Medium confidence answer")
    else:
        notes.append("Low confidence - review recommended")

    state["verification_notes"] = notes

    logger.info(f"Overall confidence: {overall:.2f}, verified: {state['is_verified']}")
    return state


class QualityWorkflow:
    """
    LangGraph workflow for answer quality verification.

    Runs factuality, citation, and completeness checks on generated answers,
    producing confidence scores.
    """

    def __init__(self):
        self.graph = self._build_graph()
        logger.info("QualityWorkflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(QualityState)

        workflow.add_node("factuality_check", check_factuality)
        workflow.add_node("citation_check", check_citations)
        workflow.add_node("completeness_check", check_completeness)
        workflow.add_node("calculate_overall", calculate_overall)

        workflow.set_entry_point("factuality_check")

        workflow.add_edge("factuality_check", "citation_check")
        workflow.add_edge("citation_check", "completeness_check")
        workflow.add_edge("completeness_check", "calculate_overall")
        workflow.add_edge("calculate_overall", END)

        return workflow.compile()

    def verify_answer(
        self, answer: str, question: str, chunks: List[Dict[str, Any]]
    ) -> VerifiedAnswer:
        """Run quality verification on an answer."""
        initial_state: QualityState = {
            "answer": answer,
            "question": question,
            "chunks": chunks,
            "factuality_score": 0.0,
            "citation_score": 0.0,
            "completeness_score": 0.0,
            "overall_confidence": 0.0,
            "verification_notes": [],
            "is_verified": False,
        }

        logger.info("Running quality verification workflow")
        final_state = self.graph.invoke(initial_state)

        sources = []
        for chunk in chunks:
            sources.append(
                SourceCitation(
                    document_name=chunk.get("document_name", ""),
                    page_number=chunk.get("page_number", 0),
                    chunk_id=chunk.get("chunk_id", ""),
                    relevance_score=chunk.get("rerank_score", chunk.get("score", 0.0)),
                )
            )

        confidence = ConfidenceScores(
            factuality=final_state["factuality_score"],
            citations=final_state["citation_score"],
            completeness=final_state["completeness_score"],
            overall=final_state["overall_confidence"],
        )

        return VerifiedAnswer(
            answer=answer,
            sources=sources,
            confidence=confidence,
            verification_notes=final_state["verification_notes"],
            is_verified=final_state["is_verified"],
        )


def get_quality_workflow() -> QualityWorkflow:
    """Get a QualityWorkflow instance."""
    return QualityWorkflow()


def verify_answer(
    answer: str, question: str, chunks: List[Dict[str, Any]]
) -> VerifiedAnswer:
    """Convenience function to verify an answer."""
    workflow = get_quality_workflow()
    return workflow.verify_answer(answer, question, chunks)
