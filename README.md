# 🔍 SourceGrounded-RAG

**A production-quality Retrieval-Augmented Generation system for multi-document Q&A — with hybrid retrieval, optional reranking, source-grounded answers, and a developer observability panel.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-Generation-4285F4?logo=google&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D)
![BM25](https://img.shields.io/badge/BM25-Keyword_Search-2d7a76)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)

---

## 💡 What is RAG?

**Retrieval-Augmented Generation (RAG)** prevents LLM hallucination by grounding every answer in your documents:

1. **Store** — Documents are chunked and converted into searchable vectors + a keyword index
2. **Retrieve** — At query time, relevant chunks are found via hybrid search (semantic + keyword)
3. **Generate** — The LLM answers using *only* the retrieved context, citing sources inline

---

## 🏗️ Architecture

```
Documents (PDF / TXT / Markdown)
        ↓
  Document Loader  (per file type, extensible registry)
        ↓
  Preprocessor     (header/footer strip, whitespace normalization, table detection)
        ↓
  Token-Aware Recursive Chunker  (tiktoken · page-aware · configurable size/overlap)
        ↓
  ┌─────────────────────────────────────┐
  │  HF Embeddings (all-MiniLM-L6-v2)  │──▶  Qdrant Vector Store
  │  BM25 Tokenizer (rank-bm25)        │──▶  BM25 In-Process Index
  └─────────────────────────────────────┘
        ↓
  Query → [ Vector Search ]  +  [ BM25 Search ]  (independent top-k each)
        ↓
  Reciprocal Rank Fusion  (rank-position merging, no score normalization)
        ↓
  [Optional] Cross-Encoder Reranker  (ms-marco-MiniLM-L-6-v2 · toggleable)
        ↓
  Refusal Check  (vector_score threshold · minimum chunk count)
        ↓
  Gemini  →  Source-Grounded Answer  +  Inline Citations
        ↓
  Streamlit UI  +  Developer Observability Panel
```

### Pipeline Stage Breakdown

| Stage | What Happens | Technology |
|-------|-------------|------------|
| **Document Loading** | Dispatch by file type; PDF returns per-page dicts with page numbers | `pypdf`, stdlib |
| **Preprocessing** | Strip repeated headers/footers; normalize whitespace; detect tables | Custom heuristics |
| **Chunking** | Recursive split on paragraph/sentence/word boundaries; token-counted | `tiktoken` |
| **Embedding** | 384-dim dense vectors capturing semantic meaning | HF API · `all-MiniLM-L6-v2` |
| **BM25 Index** | Bag-of-words keyword index built at ingestion time | `rank-bm25` |
| **Vector Search** | Cosine similarity search over all ingested chunks | `qdrant-client` |
| **BM25 Search** | Term-frequency keyword ranking over same corpus | `rank-bm25` |
| **RRF Fusion** | Rank-position merging — robust to score scale differences | Custom (Cormack 2009) |
| **Reranking** | Cross-encoder models full query×chunk interaction | `sentence-transformers` |
| **Generation** | Grounded generation with inline citations; refusal on low confidence | Google Gemini API |

---

## ✨ Features

- 📄 **Multi-Document Support** — Upload PDFs, TXT, and Markdown simultaneously; retrieval works across all
- 🧩 **Recursive Chunking** — Splits on paragraph → sentence → word boundaries (not raw byte offsets)
- 🔢 **Token-Aware Sizing** — `tiktoken` counts exact tokens; no character-approximation errors
- 📊 **Table Preservation** — Detected tables are never split mid-row (`is_table=True` metadata flag)
- 🔷 **Vector Search** — Semantic similarity via Qdrant cosine search
- 🔶 **BM25 Keyword Search** — Exact-match ranking complementing semantic search
- ⚡ **Hybrid Retrieval (RRF)** — Reciprocal Rank Fusion merges both result lists without score normalization
- 🎯 **Optional Reranking** — Cross-encoder reranker for final precision boost (toggleable via env)
- 🛡️ **Source-Grounded Answers** — LLM instructed to cite `[Source: file.pdf, Page 3]` inline
- ⛔ **Refusal Logic** — Refuses to answer when retrieval confidence is below threshold
- 🔬 **Observability Panel** — Developer panel showing vector scores, BM25 scores, reranker scores, timing, full chunk metadata, and the exact context sent to the LLM

---

## 🛠️ Real Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Frontend** | Streamlit | Multi-doc upload, observability panel |
| **Embeddings** | HF Inference API · `all-MiniLM-L6-v2` | 384-dim; remote API call |
| **Keyword Search** | `rank-bm25` · BM25Okapi | Built at ingestion time |
| **Vector DB** | Qdrant (in-memory) | Configurable disk mode via `QDRANT_MODE` |
| **Hybrid Fusion** | Reciprocal Rank Fusion | k=60 smoothing (Cormack et al. 2009) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers; lazy-loaded |
| **Generation** | Google Gemini API | Configurable model via `GENERATION_MODEL` |
| **Token Counting** | `tiktoken` · cl100k_base | Model-agnostic proxy for Gemini |
| **PDF Parsing** | `pypdf` | Page-level text extraction |

> **Not used:** OpenAI API, GPT-4o-mini, text-embedding-3-small, LangChain, LlamaIndex

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- A [Hugging Face Access Token](https://huggingface.co/settings/tokens) (read access)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SourceGrounded-RAG.git
cd SourceGrounded-RAG

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env — paste your GEMINI_API_KEY and HF_TOKEN

# 5. Run the app
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**.

### Enabling the Reranker

```bash
# In your .env file:
ENABLE_RERANKER=true
RERANK_TOP_N=5
```

The cross-encoder model (~85MB) downloads automatically on first use.

---

## 📁 Project Structure

```
SourceGrounded-RAG/
│
├── streamlit_app.py          # Frontend — multi-doc upload, Q&A, observability panel
├── rag_pipeline.py           # Orchestrator — connects all pipeline stages
│
├── ingestion/
│   ├── loaders.py            # PDF, TXT, MD loaders (extensible registry)
│   ├── preprocessor.py       # Header/footer strip, whitespace, table detection
│   └── chunker.py            # Token-aware recursive chunker (tiktoken)
│
├── retrieval/
│   ├── embedder.py           # HF Inference API wrapper (all-MiniLM-L6-v2)
│   ├── vector_store.py       # Qdrant operations with full chunk metadata
│   ├── bm25_store.py         # BM25 index — built at ingestion, singleton
│   ├── hybrid.py             # Reciprocal Rank Fusion merger
│   └── reranker.py           # Cross-encoder reranker (lazy-loaded)
│
├── generation/
│   └── generator.py          # Grounded prompting, refusal logic
│
├── requirements.txt
├── .env.example              # All configurable parameters with explanations
└── README.md
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (required) |
| `GEMINI_API_KEY` | — | Gemini API key (required) |
| `CHUNK_SIZE_TOKENS` | `300` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Token overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidates from each retriever before fusion |
| `ENABLE_RERANKER` | `false` | Enable cross-encoder reranking |
| `RERANK_TOP_N` | `5` | Chunks to keep after reranking |
| `MIN_RETRIEVAL_SCORE` | `0.30` | Minimum vector cosine score to answer (0–1) |
| `MIN_CHUNKS_REQUIRED` | `2` | Minimum chunks needed to attempt an answer |
| `GENERATION_MODEL` | `gemini-3.1-flash-lite` | Gemini model for generation |
| `QDRANT_MODE` | `memory` | `memory` or `disk` |
| `QDRANT_PATH` | `./qdrant_data` | Disk path (only when `QDRANT_MODE=disk`) |





---

## 📝 License

MIT
