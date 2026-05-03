"""
vector_store.py — Vector Database Module

Handles storing and searching document embeddings using Qdrant.

Key concept:
  A vector database stores embeddings (numerical representations of text)
  and lets you find the most similar ones to a query using cosine similarity.
  Think of it as "search by meaning" instead of "search by keyword."

We use Qdrant in local file mode — no external server needed.
Data persists in the ./qdrant_data folder.
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- Configuration ---
QDRANT_PATH = "./qdrant_data"       # Local file storage (no server needed)
COLLECTION_NAME = "pdf_chunks_hf"   # Separate from any old OpenAI 1536-dim data
VECTOR_DIM = 384                    # all-MiniLM-L6-v2 outputs 384-dimensional vectors

# --- Shared Client (created once, reused everywhere) ---
_client = QdrantClient(path=QDRANT_PATH)


def _ensure_collection():
    """Create the collection if it doesn't exist yet."""
    if not _client.collection_exists(COLLECTION_NAME):
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,  # Cosine similarity for semantic search
            ),
        )


def reset_collection():
    """
    Delete and recreate the collection.

    This is called before ingesting a new PDF so that old chunks
    don't mix with the new document. Keeps this app simple as a
    single-document demo.
    """
    if _client.collection_exists(COLLECTION_NAME):
        _client.delete_collection(COLLECTION_NAME)

    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )


def add_documents(chunks: list[str], vectors: list[list[float]], source: str) -> int:
    """
    Store document chunks and their embeddings in Qdrant.

    Each chunk is stored as a "point" with:
      - A unique ID
      - The embedding vector
      - Metadata (original text, source filename, chunk index)

    Args:
        chunks: List of text chunks
        vectors: Corresponding embedding vectors
        source: Source filename (for attribution)

    Returns:
        Number of chunks stored
    """
    _ensure_collection()

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "source": source,
                "chunk_index": i,
            },
        )
        points.append(point)

    _client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the most similar document chunks to a query vector.

    Uses cosine similarity — scores closer to 1.0 mean more
    semantically similar to the query.

    Args:
        query_vector: The embedding of the user's question
        top_k: Number of results to return

    Returns:
        List of dicts with 'text', 'source', and 'score' keys
    """
    _ensure_collection()

    # qdrant-client v1.7+ replaced .search() with .query_points()
    response = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    matches = []
    for result in response.points:
        matches.append({
            "text": result.payload.get("text", ""),
            "source": result.payload.get("source", ""),
            "score": result.score,
        })

    return matches
