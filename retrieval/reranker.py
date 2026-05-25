"""
retrieval/reranker.py — Cross-Encoder Reranker

A cross-encoder reads the query and each candidate chunk *together* in a
single forward pass, allowing it to model query–document interactions that
bi-encoders (like MiniLM embeddings) cannot capture. This catches nuanced
relevance signals that survive retrieval but are missed by embedding cosine
similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fine-tuned on MS MARCO passage ranking — well-suited for QA retrieval.
  - Outputs raw logits typically in the range -10 to +10.
  - IMPORTANT: These are NOT cosine similarities. Do NOT use reranker scores
    as retrieval confidence thresholds. Use vector_score or rrf_score instead.

Toggle: ENABLE_RERANKER=true/false (default: false)
  - Model is lazy-loaded on first use to avoid ~85MB download when disabled.

Config:
  ENABLE_RERANKER  — true/false (default: false)
  RERANK_TOP_N     — chunks to keep after reranking (default: 5)
"""

import os
from dotenv import load_dotenv

load_dotenv()

ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))

_model = None  # Lazy-loaded — only instantiated if reranker is enabled.


def _load_model():
    """Load the cross-encoder model on first use (lazy initialization)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Cross-encoder scores are raw logits — NOT cosine similarities or probabilities.
    Higher logit = more relevant to the query. These scores are appropriate for
    *ordering* candidates but must not be compared against cosine-similarity
    thresholds for refusal decisions.

    Args:
        query:      The user's question.
        candidates: Candidate chunks from hybrid retrieval.
        top_n:      Number of top chunks to return after reranking.

    Returns:
        Reranked list (length ≤ top_n) with 'reranker_score' (float logit) added.
    """
    if not candidates:
        return []

    model = _load_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs).tolist()

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {**chunk, "reranker_score": round(float(score), 4)}
        for chunk, score in scored
    ]
