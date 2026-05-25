"""
rag_pipeline.py — Core Pipeline Orchestrator

Connects all components:

  INGESTION:
    File → Loader → Preprocessor → Chunker → Embedder → Vector Store + BM25 Index

  QUERYING:
    Question → Embed → Vector Search + BM25 Search → RRF Fusion
             → [Optional Reranker] → Grounded Generation → Answer

Two public functions:
  ingest_document() — process one document and add it to the indexes
  ask()             — answer a question using all indexed documents
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

from ingestion.loaders import load_document
from ingestion.preprocessor import preprocess_pages
from ingestion.chunker import chunk_pages
from retrieval.embedder import embed
from retrieval.vector_store import (
    add_documents as vs_add,
    search as vs_search,
    reset_collection,
)
from retrieval.bm25_store import get_store as get_bm25
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import rerank, ENABLE_RERANKER, RERANK_TOP_N
from generation.generator import generate_answer

load_dotenv()

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))


def ingest_document(
    file_path: str,
    source_name: str | None = None,
    reset: bool = True,
) -> int:
    """
    Full ingestion pipeline for one document.

    Steps:
      1. Load document pages via the appropriate loader (PDF/TXT/MD)
      2. Preprocess: strip headers/footers, normalize whitespace, detect tables
      3. Chunk: token-aware recursive chunking with page boundaries respected
      4. Embed: generate HF embeddings for all chunks
      5. Index: store in Qdrant (vector) and BM25 (keyword) simultaneously

    Args:
        file_path:   Path to the document on disk.
        source_name: Display name for citations (defaults to filename).
        reset:       If True, clear existing indexes before ingesting.
                     Set to False when adding subsequent documents in a batch.

    Returns:
        Number of chunks ingested (0 if the document was empty after preprocessing).
    """
    if source_name is None:
        source_name = Path(file_path).name

    doc_type = Path(file_path).suffix.lstrip(".").lower()

    if reset:
        reset_collection()
        get_bm25().reset()

    # Step 1: Load
    pages = load_document(file_path)
    if not pages:
        return 0

    # Step 2: Preprocess
    pages = preprocess_pages(pages)
    if not pages:
        return 0

    # Step 3: Chunk
    chunks = chunk_pages(pages, source=source_name, doc_type=doc_type)
    if not chunks:
        return 0

    # Step 4: Embed (batch call to HF API)
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    # Step 5: Index into vector store and BM25 simultaneously
    vs_add(chunks, vectors)
    get_bm25().add_documents(chunks)

    return len(chunks)


def ask(question: str, top_k: int = RETRIEVAL_TOP_K) -> dict:
    """
    Full RAG query pipeline.

    Steps:
      1. Embed the question via HF API
      2. Vector search in Qdrant (top_k results)
      3. BM25 keyword search (top_k results)
      4. Reciprocal Rank Fusion — merge and deduplicate both result lists
      5. [Optional] Cross-encoder reranking (if ENABLE_RERANKER=true)
      6. Grounded generation with refusal logic

    Args:
        question: The user's question.
        top_k:    Candidate chunks to retrieve from each system before fusion.

    Returns:
        Dict with:
          answer          (str):        Final answer or refusal message.
          refused         (bool):       True if refusal was triggered.
          refusal_reason  (str):        Why refusal was triggered (empty if not).
          sources         (list[str]):  Unique source documents.
          chunks          (list[dict]): Final chunks passed to LLM, with all scores.
          context_block   (str):        Exact context sent to the LLM.
          retrieval_ms    (float):      Time spent in retrieval (ms).
          vector_results  (list[dict]): Raw vector search results (for observability).
          bm25_results    (list[dict]): Raw BM25 results (for observability).
          reranker_enabled (bool):      Whether the reranker was active.
    """
    t0 = time.perf_counter()

    # Step 1: Embed the query.
    query_vector = embed([question])[0]

    # Step 2: Vector similarity search.
    vector_results = vs_search(query_vector, top_k=top_k)

    # Step 3: BM25 keyword search.
    bm25_results = get_bm25().search(question, top_k=top_k)

    # Step 4: Reciprocal Rank Fusion — combine and deduplicate both result lists.
    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)

    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Step 5: Optionally rerank with cross-encoder.
    if ENABLE_RERANKER and fused:
        chunks = rerank(question, fused, top_n=RERANK_TOP_N)
    else:
        chunks = fused

    # Step 6: Grounded generation with refusal check.
    result = generate_answer(question, chunks)

    # Attach debug metadata for the observability panel.
    result["chunks"] = chunks
    result["retrieval_ms"] = round(retrieval_ms, 1)
    result["vector_results"] = vector_results
    result["bm25_results"] = bm25_results
    result["reranker_enabled"] = ENABLE_RERANKER

    return result
