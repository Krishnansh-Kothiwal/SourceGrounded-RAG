"""
generation/generator.py — Grounded Answer Generator

Wraps Gemini generation with:
  - Source-grounded prompting: LLM is instructed to cite retrieved chunks inline.
  - Refusal logic: returns a standard refusal when retrieval confidence is low.

Refusal thresholds (configurable via .env):
  MIN_RETRIEVAL_SCORE  — minimum vector_score of the top result (default: 0.30)
                         Uses cosine similarity (0–1 range). Falls back to
                         rrf_score if vector_score is unavailable.
  MIN_CHUNKS_REQUIRED  — minimum number of retrieved chunks (default: 2)

IMPORTANT — score semantics:
  Refusal uses vector_score (cosine similarity, 0–1) or rrf_score.
  It does NOT use reranker_score. Cross-encoder scores are raw logits (≈ -10
  to +10) with no meaningful threshold in the 0–1 cosine similarity range.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.30"))
MIN_CHUNKS_REQUIRED: int = int(os.getenv("MIN_CHUNKS_REQUIRED", "2"))

REFUSAL_MESSAGE = (
    "I cannot answer this reliably from the retrieved documents."
)

SYSTEM_INSTRUCTION = (
    "You are a precise document analyst. Answer questions using ONLY the context "
    "provided. For each factual claim, cite the source using the format "
    "[Source: <filename>, Page <n>] or [Source: <filename>] when no page is "
    "available. Do not draw on any knowledge outside the provided context. "
    f'If the context is insufficient, say exactly: "{REFUSAL_MESSAGE}"'
)

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _should_refuse(chunks: list[dict]) -> tuple[bool, str]:
    """
    Decide whether to refuse answering based on retrieval quality.

    Checks (in order):
      1. Too few chunks retrieved (< MIN_CHUNKS_REQUIRED)
      2. Top chunk's vector_score below MIN_RETRIEVAL_SCORE

    Score selection logic:
      - Prefers vector_score (cosine similarity, 0–1 range from Qdrant).
      - Falls back to rrf_score if vector_score is missing (e.g. chunk only
        appeared in BM25 results).
      - Never uses reranker_score — cross-encoder logits are incomparable
        to a cosine similarity threshold.

    Returns:
        (should_refuse: bool, reason: str)
    """
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return True, (
            f"Only {len(chunks)} chunk(s) retrieved "
            f"(minimum required: {MIN_CHUNKS_REQUIRED})"
        )

    top = chunks[0]
    score = top.get("vector_score") or top.get("rrf_score") or 0.0

    if score < MIN_RETRIEVAL_SCORE:
        return True, (
            f"Top retrieval score {score:.3f} is below the confidence "
            f"threshold {MIN_RETRIEVAL_SCORE}"
        )

    return False, ""


def _format_context_block(chunks: list[dict]) -> str:
    """
    Build the context string passed to the LLM.

    Each chunk is prefixed with its source citation so the LLM can reproduce
    accurate inline citations in its answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")
        citation = (
            f"[Source: {source}, Page {page}]" if page
            else f"[Source: {source}]"
        )
        parts.append(f"--- Chunk {i} {citation} ---\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Generate a source-grounded answer from retrieved chunks.

    Args:
        question: The user's question.
        chunks:   Ranked chunk dicts from retrieval or reranking.

    Returns:
        Dict with:
          answer          (str):       LLM answer or refusal message.
          refused         (bool):      True if refusal was triggered.
          refusal_reason  (str):       Human-readable reason (empty if not refused).
          context_block   (str):       Exact context string sent to the LLM.
          sources         (list[str]): Unique source document names cited.
    """
    should_refuse, reason = _should_refuse(chunks)

    if should_refuse:
        return {
            "answer":         REFUSAL_MESSAGE,
            "refused":        True,
            "refusal_reason": reason,
            "context_block":  "",
            "sources":        [],
        }

    context_block = _format_context_block(chunks)

    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite each claim with [Source: ...] inline):"
    )

    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,        # Low temperature for factual, grounded answers.
            max_output_tokens=1024,
        ),
    )

    sources = list({
        f"{c['source']} (Page {c['page']})" if c.get("page") else c.get("source")
        for c in chunks if c.get("source")
    })

    return {
        "answer":         response.text.strip(),
        "refused":        False,
        "refusal_reason": "",
        "context_block":  context_block,
        "sources":        sources,
    }
