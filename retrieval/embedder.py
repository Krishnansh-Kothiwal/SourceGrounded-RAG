"""
retrieval/embedder.py — Hugging Face Embeddings

Calls the HuggingFace Inference API to generate 384-dimensional sentence
embeddings using sentence-transformers/all-MiniLM-L6-v2.

All embedding calls in this codebase go through embed() — the single
public function — so swapping the embedding backend only requires changing
this file.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


def embed(texts: list[str]) -> list[list[float]]:
    """
    Convert text strings into 384-dimensional embedding vectors via HF API.

    Args:
        texts: List of text strings to embed. Batching is handled by the API.

    Returns:
        List of embedding vectors — each a list of 384 floats.

    Raises:
        Exception: If the HF API returns a non-200 status code.
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json",
    }

    response = requests.post(HF_API_URL, headers=headers, json={"inputs": texts})

    if response.status_code != 200:
        raise Exception(
            f"HuggingFace Embedding API error {response.status_code}: {response.text}"
        )

    data = response.json()

    # The API returns a single vector (list of floats) for a single-item input.
    # Normalize to always return a list of vectors.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], float):
        return [data]
    return data
