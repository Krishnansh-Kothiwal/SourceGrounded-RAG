"""
rag_pipeline.py — Core RAG Pipeline

This is the brain of the application. It connects all the pieces:

  INGESTION:  PDF → chunks → embeddings → vector store
  QUERYING:   question → embedding → search → context → LLM → answer

Two main functions:
  - ingest_document(): Process and store a PDF
  - ask(): Answer a question using stored documents
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ingestion import load_pdf, chunk_text, get_embeddings
from vector_store import add_documents, search, reset_collection

load_dotenv()

# --- Configuration ---
# Embeddings: Hugging Face (all-MiniLM-L6-v2)
# Generation: Google Gemma 4 (via Gemini API)
GENERATION_MODEL = "gemma-4-26b-a4b-it"  # Gemma 4 MoE — 26B total / 4B active params

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer only using the provided context. "
    "If the answer is not in the context, say you do not know."
)

# Configure Gemini API client (hosts Gemma models too)
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def ingest_document(file_path: str, source_name: str = None) -> int:
    """
    Full document ingestion pipeline:
      1. Read PDF → extract text
      2. Split text into overlapping chunks
      3. Generate an embedding for each chunk
      4. Store chunks + embeddings in vector database

    Args:
        file_path: Path to the PDF file
        source_name: Display name for the source (defaults to file_path)

    Returns:
        Number of chunks ingested
    """
    if source_name is None:
        source_name = file_path

    # Clear old chunks — this is a single-document demo
    reset_collection()

    # Step 1: Extract text from PDF
    text = load_pdf(file_path)

    if not text.strip():
        return 0

    # Step 2: Split into chunks
    chunks = chunk_text(text)

    if not chunks:
        return 0

    # Step 3: Generate embeddings via Hugging Face
    vectors = get_embeddings(chunks)

    # Step 4: Store in Qdrant
    count = add_documents(chunks, vectors, source_name)

    return count


def ask(question: str, top_k: int = 5) -> dict:
    """
    Full RAG query pipeline:
      1. Convert the question into an embedding
      2. Search the vector store for similar chunks
      3. Assemble retrieved chunks into a context block
      4. Send context + question to Gemini for grounded generation
      5. Return the answer

    Args:
        question: The user's question
        top_k: Number of chunks to retrieve (default 5)

    Returns:
        Dict with 'answer', 'sources', and 'contexts' keys
    """
    # Step 1: Embed the question
    query_vector = get_embeddings([question])[0]

    # Step 2: Search for relevant chunks
    results = search(query_vector, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload a PDF first.",
            "sources": [],
            "contexts": [],
        }

    # Step 3: Build context from retrieved chunks
    contexts = [r["text"] for r in results]
    sources = list(set(r["source"] for r in results))
    context_block = "\n\n---\n\n".join(contexts)

    # Step 4: Build the grounded prompt
    prompt = (
        f"Answer the question based only on the context below.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    # Step 5: Call Gemma 4 for generation
    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=512,
        ),
    )
    answer = response.text.strip()

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }
