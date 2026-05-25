"""
retrieval/hybrid.py — Reciprocal Rank Fusion (RRF)

Merges ranked result lists from vector search and BM25 into a single
unified ranking using Reciprocal Rank Fusion.

Why RRF over score normalization:
  - Vector cosine scores (0–1) and BM25 scores (0–∞) live in completely
    different numerical ranges with different statistical distributions.
    Normalizing them requires knowing the global min/max per query, which
    changes with every query and can be dominated by outliers.
  - RRF only cares about rank *position*, not score magnitude. A document
    at rank 3 contributes 1/(60+3) = 0.0159 regardless of its raw score.
    This makes RRF robust and parameter-free across retrieval systems.
  - A document appearing in both result lists receives contributions from
    both, naturally boosting documents that are relevant by both measures.
  - The smoothing constant k=60 is the standard value from Cormack et al.
    (2009), validated empirically across TREC benchmarks.

RRF formula:  score(d) = Σ_r  1 / (k + rank_r(d))
"""

from collections import defaultdict

RRF_K = 60  # Cormack et al. 2009 smoothing constant


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Fuse vector and BM25 result lists using Reciprocal Rank Fusion.

    Args:
        vector_results: Ranked results from vector search.
                        Each dict includes chunk metadata + 'vector_score'.
        bm25_results:   Ranked results from BM25 search.
                        Each dict includes chunk metadata + 'bm25_score'.
        top_k:          Number of results to return after fusion.

    Returns:
        Merged list of chunk dicts sorted by descending RRF score.
        Each result includes:
          - All original chunk metadata
          - 'vector_score': cosine similarity (None if not in vector results)
          - 'bm25_score':   BM25 score (None if not in BM25 results)
          - 'rrf_score':    combined RRF score (float)
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    metadata: dict[str, dict] = {}

    # Contribute RRF score from vector search ranking.
    for rank, result in enumerate(vector_results, start=1):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] += 1.0 / (RRF_K + rank)
        if key not in metadata:
            metadata[key] = {**result, "bm25_score": None}

    # Contribute RRF score from BM25 ranking.
    for rank, result in enumerate(bm25_results, start=1):
        key = f"{result['source']}::{result['chunk_index']}"
        rrf_scores[key] += 1.0 / (RRF_K + rank)
        if key not in metadata:
            # Document appears only in BM25 results, not in vector results.
            metadata[key] = {**result, "vector_score": None}
        else:
            # Merge BM25 score into the existing metadata entry.
            metadata[key]["bm25_score"] = result.get("bm25_score")

    # Sort by descending RRF score and return top_k results.
    sorted_keys = sorted(
        rrf_scores, key=lambda k: rrf_scores[k], reverse=True
    )[:top_k]

    return [
        {**metadata[key], "rrf_score": round(rrf_scores[key], 6)}
        for key in sorted_keys
    ]
