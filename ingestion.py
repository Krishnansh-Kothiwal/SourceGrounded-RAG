"""
ingestion.py — Document Ingestion Module

Handles the first half of the RAG pipeline:
  PDF File → Text Extraction → Text Chunking → Embedding Generation

Key concepts:
  - We extract raw text from PDFs
  - We split text into small overlapping "chunks" (why? because embeddings
    work better on focused passages, and LLMs have token limits)
  - We convert each chunk into a numerical vector (embedding) that captures
    its *meaning*, so semantically similar text = nearby vectors
"""

import os
import requests
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Using Hugging Face Inference API for embeddings
# all-MiniLM-L6-v2 is a lightweight, fast, and high-quality sentence embedding model
# It outputs 384-dimensional vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        A single string containing all text from every page
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.

    Why chunk?
      - Embeddings work better on focused, smaller passages
      - LLMs have limited context windows
      - Overlap ensures we don't lose meaning at chunk boundaries

    Args:
        text: The full document text
        chunk_size: Max characters per chunk (default 1000)
        chunk_overlap: Characters to overlap between chunks (default 200)

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        # Move forward, overlapping with the previous chunk
        start += chunk_size - chunk_overlap

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert text strings into 384-dimensional embedding vectors using
    Hugging Face Inference API (sentence-transformers/all-MiniLM-L6-v2).

    An embedding is a list of numbers (a vector) that represents the
    *meaning* of the text. Similar meanings → nearby vectors.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of 384 floats)
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": texts},
    )

    if response.status_code != 200:
        raise Exception(
            f"Hugging Face Embedding API error {response.status_code}: {response.text}"
        )

    data = response.json()

    # Normalize response: single embedding → wrap in list; list of embeddings → return as-is
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], float):
        return [data]  # Single embedding returned, wrap it
    return data  # Already a list of embeddings
