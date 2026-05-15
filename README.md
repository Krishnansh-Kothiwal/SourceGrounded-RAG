# 🔍 SourceGrounded-RAG

**A Retrieval-Augmented Generation application that lets you upload PDFs and ask questions — with answers grounded in your actual documents.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemma--4-26B-4285F4?logo=google&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-FFD21E?logo=huggingface&logoColor=black)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)

---

## 💡 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that makes AI answers more accurate by grounding them in your data:

1. **Store** — Your documents are split into chunks and converted into searchable vectors
2. **Retrieve** — When you ask a question, the most relevant chunks are found via semantic search
3. **Generate** — An LLM generates an answer using *only* the retrieved context

This prevents AI hallucination by anchoring every answer in your actual documents.

---

## 🏗️ Architecture

```
┌─────────────────────── INGESTION PIPELINE ───────────────────────┐
│                                                                   │
│   📄 PDF Upload → Text Extraction → Chunking → Embeddings → Qdrant │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────────── QUERY PIPELINE ───────────────────────────┐
│                                                                   │
│   ❓ Question → Embedding → Similarity Search → Context + LLM → ✅ Answer │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Pipeline Breakdown

| Stage | What Happens | Technology |
|-------|-------------|------------|
| **PDF Reading** | Extract raw text from uploaded PDFs | `pypdf` |
| **Chunking** | Split text into 1000-char overlapping segments | Custom splitter |
| **Embedding** | Convert text chunks into 384-dim vectors | Hugging Face `all-MiniLM-L6-v2` |
| **Storage** | Temporary vector storage for retrieval | Qdrant (In-Memory) |
| **Retrieval** | Find top-K most similar chunks to the query | Cosine similarity search |
| **Generation** | LLM answers using only retrieved context | Google `Gemma-4-26B-A4B` |

---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF to build a searchable knowledge base
- 🧩 **Overlapping Chunking** — Smart text splitting preserves context across boundaries
- 🔢 **Semantic Embeddings** — Text converted to meaning-aware vectors
- 🗄️ **In-Memory Search** — Qdrant runs in-memory for instant, fresh results (no stale data)
- 🤖 **Grounded Answers** — Gemma 4 answers using *only* your documents
- 📎 **Source Attribution** — See which chunks informed each answer

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Frontend** | Streamlit | Simple, Python-native UI |
| **Embeddings** | Hugging Face API | Lightweight, fast, high-quality |
| **Vector DB** | Qdrant (In-Memory) | Fast, no lock issues, fresh state |
| **LLM** | Google Gemma 4 | State-of-the-art MoE reasoning |
| **PDF Parser** | pypdf | Lightweight, no dependencies |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- A [Hugging Face Access Token](https://huggingface.co/settings/tokens)

### Setup (4 steps)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SourceGrounded-RAG.git
cd SourceGrounded-RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# Edit .env and paste your Gemini API key and Hugging Face token

# 4. Run the app
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501** — upload a PDF and start asking questions!

---

## 📁 Project Structure

```
SourceGrounded-RAG/
├── streamlit_app.py    # Frontend UI — upload PDFs and ask questions
├── rag_pipeline.py     # Core pipeline — connects ingestion and querying
├── ingestion.py        # PDF loading, text chunking, embedding generation
├── vector_store.py     # Qdrant operations — in-memory storage & search
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore          # Files excluded from version control
└── README.md           # You are here
```

---

## 🔍 How It Works

### 1. Document Ingestion
When you upload a PDF, the app:
- **Extracts text** from every page using `pypdf`
- **Chunks the text** into ~1000-character segments with 200-character overlap (so no context is lost at boundaries)
- **Generates embeddings** — each chunk is sent to Hugging Face's inference API, which returns a 384-dimensional vector capturing its semantic meaning
- **Stores in Qdrant** — vectors are saved in an **in-memory** collection (guaranteed fresh on every restart)

### 2. Question Answering
When you ask a question:
- **Embeds the question** using the same embedding model
- **Searches Qdrant** for the top-K chunks whose vectors are most similar (cosine similarity)
- **Builds a prompt** with the retrieved chunks as context
- **Sends to Google Gemma 4** with instructions to answer *only* from the provided context
- **Returns the answer** with source attribution

---

## ⚙️ Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Your Google Gemini API key |
| `HF_TOKEN` | ✅ | Your Hugging Face access token |

All configuration is in `.env`. See `.env.example` for the template.

---

## 🗺️ Future Improvements

- [ ] Support for multiple file formats (DOCX, TXT, Markdown)
- [ ] Chat history and follow-up questions
- [ ] Hybrid search (keyword + semantic)
- [ ] Local LLM support via Ollama

---

## 📝 License

MIT
