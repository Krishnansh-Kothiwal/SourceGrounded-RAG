"""
streamlit_app.py — Frontend UI

A clean Streamlit interface for the SourceGrounded-RAG application.
Two sections:
  1. Upload PDFs to build your knowledge base
  2. Ask questions and get grounded answers
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SourceGrounded RAG",
    page_icon="🔍",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom Styling
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide the Streamlit running indicator and toolbar (Deploy) */
    [data-testid="stStatusWidget"], [data-testid="stToolbar"] {
        display: none !important;
    }
    
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #45A29E, #66FCF1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #C5C6C7;
        opacity: 0.8;
        font-size: 1.05rem;
    }
    .answer-box {
        background: #1F2833;
        border: 1px solid #45A29E;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .source-tag {
        display: inline-block;
        background: #0B0C10;
        border: 1px solid #45A29E;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.82rem;
        color: #C5C6C7;
        margin: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🔍 SourceGrounded RAG</h1>
        <p>Upload PDFs · Ask questions · Get answers grounded in your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# API Key Check
# ──────────────────────────────────────────────
if not os.getenv("HF_TOKEN"):
    st.error(
        "⚠️ **Hugging Face API token not found.**  \n"
        "Add to your `.env` file:  \n"
        "`HF_TOKEN=hf_your-token-here`"
    )
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.error(
        "⚠️ **Gemini API key not found.**  \n"
        "Add to your `.env` file:  \n"
        "`GEMINI_API_KEY=your-gemini-api-key-here`  \n"
        "Get yours at: https://aistudio.google.com/app/apikey"
    )
    st.stop()

# Import after env checks so all clients initialize correctly
from rag_pipeline import ingest_document, ask  # noqa: E402

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# ──────────────────────────────────────────────
# Section 1: Document Upload
# ──────────────────────────────────────────────
st.header("📄 Upload a Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a PDF to add it to your searchable knowledge base.",
)

if uploaded_file:
    st.info(f"📎 Selected: **{uploaded_file.name}**")

    if st.button("⚙️ Process PDF", type="primary"):
        with st.spinner(f"Processing **{uploaded_file.name}**..."):
            # Save upload to a temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                count = ingest_document(tmp_path, source_name=uploaded_file.name)
                st.success(f"✅ Ingested **{count} chunks** from `{uploaded_file.name}`")
                st.session_state.ingested_files = [uploaded_file.name]
            except Exception as e:
                st.error(f"❌ Error during ingestion: {e}")
            finally:
                os.unlink(tmp_path)

# Show ingested documents
if st.session_state.ingested_files:
    st.caption(f"📚 Active document: **{st.session_state.ingested_files[0]}**")

st.divider()

# ──────────────────────────────────────────────
# Section 2: Ask a Question
# ──────────────────────────────────────────────
st.header("❓ Ask a Question")

question = st.text_input(
    "What would you like to know?",
    placeholder="e.g., What are the main findings in the report?",
)

col1, col2 = st.columns([4, 1])
with col2:
    top_k = st.number_input("Chunks", min_value=1, max_value=20, value=5, help="Number of document chunks to retrieve")

if st.button("🔍 Get Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask(question.strip(), top_k=int(top_k))

                # -- Answer --
                st.markdown("### 💡 Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # -- Sources --
                if result["sources"]:
                    src_tags = " ".join(
                        f'<span class="source-tag">📎 {s}</span>'
                        for s in result["sources"]
                    )
                    st.markdown(f"**Sources:** {src_tags}", unsafe_allow_html=True)

                # -- Retrieved context (collapsible) --
                if result["contexts"]:
                    with st.expander("Retrieved Context Chunks", expanded=False):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            display = ctx[:500] + "..." if len(ctx) > 500 else ctx
                            st.text(display)
                            if i < len(result["contexts"]):
                                st.divider()

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · HF Embeddings · Gemini · Qdrant  —  **SourceGrounded-RAG**")
