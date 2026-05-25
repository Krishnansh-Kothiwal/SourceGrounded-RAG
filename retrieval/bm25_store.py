"""
retrieval/bm25_store.py — BM25 Keyword Index

BM25 (Best Match 25) is a bag-of-words ranking function that scores documents
by term frequency weighted against inverse document frequency. It excels at
exact keyword matches that semantic embeddings sometimes miss — acronyms,
proper nouns, version numbers, and technical identifiers.

Why BM25 complements vector search:
  - Semantic embeddings capture meaning but can confuse similar concepts
    (e.g. "Python snake" and "Python programming" may have close vectors).
  - BM25 rewards exact term overlap, catching cases where keyword specificity
    matters more than conceptual similarity.
  - Hybrid retrieval using both exploits their complementary failure modes.

Design decisions:
  - Index is built at ingestion time (not query time). Building once is O(n);
    rebuilding at every query would be O(n) per query, which is wasteful.
  - rank-bm25's BM25Okapi uses IDF smoothing to avoid zero-division on rare terms.
  - The index lives in-process memory alongside Qdrant, so it has the same
    lifecycle and requires no external synchronization.
"""

import re
from rank_bm25 import BM25Okapi


class BM25Store:
    """
    In-process BM25 index over ingested document chunks.

    Thread safety: not thread-safe. Streamlit runs single-threaded per session,
    so this is acceptable for this use case.
    """

    def __init__(self):
        self._corpus: list[dict] = []       # full chunk metadata dicts
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and strip punctuation before splitting on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def reset(self):
        """Clear the index entirely. Called before a fresh ingestion batch."""
        self._corpus = []
        self._tokenized = []
        self._index = None

    def add_documents(self, chunks: list[dict]):
        """
        Extend the BM25 index with new document chunks and rebuild.

        BM25Okapi requires the complete corpus at construction time, so the
        index is rebuilt from scratch on every call. This is acceptable because
        ingestion is a one-time operation per session.

        Args:
            chunks: List of chunk dicts; must include a 'text' field plus
                    all metadata fields (source, page, chunk_index, etc.).
        """
        for chunk in chunks:
            self._corpus.append(chunk)
            self._tokenized.append(self._tokenize(chunk["text"]))

        if self._tokenized:
            self._index = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int) -> list[dict]:
        """
        Return the top_k chunks ranked by BM25 score.

        Args:
            query: Raw query string (tokenized internally).
            top_k: Number of results to return.

        Returns:
            List of chunk dicts with 'bm25_score' field added.
            Empty list if the index is empty.
        """
        if self._index is None or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {**self._corpus[idx], "bm25_score": round(float(score), 4)}
            for idx, score in ranked
        ]

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._corpus)


# Module-level singleton shared across the Streamlit session.
# Python's module cache ensures this is the same object for the process lifetime.
_store = BM25Store()


def get_store() -> BM25Store:
    """Return the shared BM25Store singleton."""
    return _store
