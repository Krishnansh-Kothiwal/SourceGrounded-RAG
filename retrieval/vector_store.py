"""
retrieval/vector_store.py — Qdrant Vector Store

Stores document chunk embeddings and enables cosine-similarity search.
Updated from the original to:
  - Accept full chunk metadata dicts (page, doc_type, is_table, token_count)
  - Return full metadata in search results (including vector_score)
  - Support multi-document ingestion (add without reset between documents)

Config:
  QDRANT_MODE=memory|disk  — in-memory (default) or persistent disk storage
  QDRANT_PATH              — disk path when QDRANT_MODE=disk (default: ./qdrant_data)
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

COLLECTION_NAME = "doc_chunks"
VECTOR_DIM = 384  # all-MiniLM-L6-v2

_MODE = os.getenv("QDRANT_MODE", "memory").lower()
_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")

# Client is initialized once at module import. In-memory is the default because
# it guarantees a clean state with no stale data from previous sessions.
if _MODE == "disk":
    _client = QdrantClient(path=_PATH)
else:
    _client = QdrantClient(":memory:")


def get_client() -> QdrantClient:
    return _client


def _ensure_collection():
    """Create the collection if it does not exist yet."""
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def reset_collection():
    """
    Drop and recreate the collection. Called before a fresh ingestion batch
    so that old chunks from previous uploads do not pollute new results.
    """
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )


def add_documents(chunks: list[dict], vectors: list[list[float]]) -> int:
    """
    Store document chunks and their embeddings in Qdrant.

    Args:
        chunks:  List of chunk dicts (must include 'text' and all metadata fields).
        vectors: Corresponding embedding vectors, same length as chunks.

    Returns:
        Number of points stored.
    """
    _ensure_collection()
    client = get_client()

    points = []
    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":        chunk["text"],
                "source":      chunk.get("source", ""),
                "page":        chunk.get("page"),
                "doc_type":    chunk.get("doc_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "is_table":    chunk.get("is_table", False),
                "token_count": chunk.get("token_count", 0),
            },
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(query_vector: list[float], top_k: int = 10) -> list[dict]:
    """
    Find the top_k most similar document chunks via cosine similarity.

    Returns:
        List of dicts with all stored metadata plus 'vector_score'
        (cosine similarity, 0–1 range where 1.0 = identical).
    """
    _ensure_collection()
    client = get_client()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        results.append({
            "text":         point.payload.get("text", ""),
            "source":       point.payload.get("source", ""),
            "page":         point.payload.get("page"),
            "doc_type":     point.payload.get("doc_type", ""),
            "chunk_index":  point.payload.get("chunk_index", 0),
            "is_table":     point.payload.get("is_table", False),
            "token_count":  point.payload.get("token_count", 0),
            "vector_score": round(point.score, 4),
        })

    return results


def collection_size() -> int:
    """Return total number of stored chunks, or 0 if collection does not exist."""
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    return client.count(collection_name=COLLECTION_NAME).count
