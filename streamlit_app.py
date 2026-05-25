"""
streamlit_app.py — SourceGrounded RAG Frontend

Features:
  - Multi-document upload (PDF, TXT, Markdown)
  - Hybrid retrieval (vector + BM25) with optional cross-encoder reranking
  - Source-grounded answers with per-chunk citations
  - Refusal display when retrieval confidence is low
  - Developer observability panel (collapsible, off by default)
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SourceGrounded RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom Styling
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0B0C10;
        color: #C5C6C7;
    }

    [data-testid="stStatusWidget"],
    [data-testid="stToolbar"] { display: none !important; }

    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1f2d3d;
    }

    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #45A29E 0%, #66FCF1 60%, #a8f7f3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #8a9bb0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #141a23 0%, #1a2232 100%);
        border: 1px solid #45A29E;
        border-left: 4px solid #66FCF1;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 0.75rem 0;
        line-height: 1.75;
        font-size: 0.97rem;
    }

    /* Refusal box */
    .refusal-box {
        background: linear-gradient(135deg, #1a1208 0%, #241a0e 100%);
        border: 1px solid #b87333;
        border-left: 4px solid #f0a500;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin: 0.75rem 0;
        color: #f0c060;
        font-size: 0.97rem;
    }
    .refusal-reason {
        font-size: 0.82rem;
        color: #a08050;
        margin-top: 0.5rem;
    }

    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #111820;
        border: 1px solid #45A29E;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        color: #66FCF1;
        margin: 0.2rem 0.15rem;
        font-weight: 500;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: #1a2232;
        border: 1px solid #2d4060;
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.75rem;
        color: #8ab4d4;
        font-family: monospace;
        margin-left: 0.4rem;
    }
    .score-badge.high { border-color: #45A29E; color: #66FCF1; }
    .score-badge.medium { border-color: #a08030; color: #d4a840; }
    .score-badge.low { border-color: #802020; color: #d46060; }
    .score-badge.table-flag {
        background: #1a1030;
        border-color: #7060c0;
        color: #b0a0f0;
    }

    /* Chunk card in observability panel */
    .chunk-card {
        background: #111820;
        border: 1px solid #1e2d40;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .chunk-meta {
        font-size: 0.78rem;
        color: #607080;
        margin-bottom: 0.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .chunk-text {
        color: #a0b0c0;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Sidebar doc list */
    .doc-item {
        background: #111820;
        border: 1px solid #1e2d40;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.83rem;
        color: #8ab4d4;
    }

    /* Timing pill */
    .timing-pill {
        display: inline-block;
        background: #0f1820;
        border: 1px solid #2d4060;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        color: #60a0d0;
        font-family: monospace;
    }

    /* Section headers */
    h2 { color: #c8d8e8 !important; }
    h3 { color: #b0c8d8 !important; font-size: 1.1rem !important; }

    /* Streamlit button overrides */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #45A29E, #2d7a76);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #66FCF1, #45A29E);
        color: #0B0C10;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(70, 162, 158, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# API Key Checks (before any imports that use them)
# ──────────────────────────────────────────────
if not os.getenv("HF_TOKEN"):
    st.error(
        "⚠️ **Hugging Face API token not found.**\n\n"
        "Add to your `.env` file:\n```\nHF_TOKEN=hf_your-token-here\n```"
    )
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.error(
        "⚠️ **Gemini API key not found.**\n\n"
        "Add to your `.env` file:\n```\nGEMINI_API_KEY=your-gemini-api-key\n```\n\n"
        "Get one at: https://aistudio.google.com/app/apikey"
    )
    st.stop()

# Import after env checks so clients initialize with valid keys.
from rag_pipeline import ingest_document, ask  # noqa: E402
from retrieval.reranker import ENABLE_RERANKER  # noqa: E402

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []   # list of {"name": str, "chunks": int}
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🔍 SourceGrounded RAG</h1>
        <p>Hybrid retrieval · Source-grounded answers · Multi-document support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — Document Library & Config
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Document Library")

    if st.session_state.ingested_files:
        for doc in st.session_state.ingested_files:
            st.markdown(
                f'<div class="doc-item">📄 <b>{doc["name"]}</b>'
                f'<br><span style="color:#506070;">{doc["chunks"]} chunks</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.markdown("## ⚙️ Retrieval Settings")

    top_k = st.slider(
        "Candidates per retriever (top-k)",
        min_value=3,
        max_value=20,
        value=int(os.getenv("RETRIEVAL_TOP_K", "10")),
        help="Number of chunks retrieved from vector search AND BM25 independently before fusion.",
    )

    st.divider()
    st.markdown("## 🔬 Pipeline Status")
    reranker_status = "✅ Enabled" if ENABLE_RERANKER else "⭕ Disabled"
    st.caption(f"Reranker: {reranker_status}")
    st.caption("Set `ENABLE_RERANKER=true` in `.env` to activate.")
    st.caption(f"Retrieval mode: Vector + BM25 → RRF")

# ──────────────────────────────────────────────
# Section 1: Document Upload
# ──────────────────────────────────────────────
st.markdown("## 📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Choose one or more documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Supported: PDF, plain text (.txt), Markdown (.md). All documents are indexed together.",
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    st.info(f"📎 Selected: **{', '.join(file_names)}**")

    if st.button("⚙️ Process & Index Documents", type="primary"):
        progress = st.progress(0, text="Preparing...")
        total = len(uploaded_files)
        new_docs = []
        errors = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                (i) / total,
                text=f"Processing **{uploaded_file.name}** ({i+1}/{total})...",
            )

            # Preserve the original file extension for the loader dispatcher.
            ext = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                # reset=True only for the first document to clear old indexes.
                count = ingest_document(
                    tmp_path,
                    source_name=uploaded_file.name,
                    reset=(i == 0),
                )
                new_docs.append({"name": uploaded_file.name, "chunks": count})
            except Exception as e:
                errors.append(f"**{uploaded_file.name}**: {e}")
            finally:
                os.unlink(tmp_path)

        progress.progress(1.0, text="Done!")

        if new_docs:
            st.session_state.ingested_files = new_docs
            total_chunks = sum(d["chunks"] for d in new_docs)
            st.success(
                f"✅ Indexed **{len(new_docs)} document(s)** → "
                f"**{total_chunks} total chunks**"
            )

        for err in errors:
            st.error(f"❌ Error — {err}")

        if new_docs:
            st.rerun()

st.divider()

# ──────────────────────────────────────────────
# Section 2: Ask a Question
# ──────────────────────────────────────────────
st.markdown("## ❓ Ask a Question")

if not st.session_state.ingested_files:
    st.info("Upload and process at least one document above to start asking questions.")
else:
    question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What are the main findings in the report?",
        label_visibility="collapsed",
    )

    col_btn, col_spacer = st.columns([2, 5])
    with col_btn:
        search_clicked = st.button(
            "🔍 Get Answer", type="primary", use_container_width=True
        )

    if search_clicked:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Running hybrid retrieval and generating answer..."):
                try:
                    result = ask(question.strip(), top_k=top_k)
                    st.session_state.last_result = {"question": question, **result}
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")
                    st.session_state.last_result = None

# ──────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────
if st.session_state.last_result:
    result = st.session_state.last_result

    st.markdown("### 💡 Answer")

    if result.get("refused"):
        st.markdown(
            f'<div class="refusal-box">'
            f'⚠️ {result["answer"]}'
            f'<div class="refusal-reason">Reason: {result.get("refusal_reason", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

    # Source attribution
    if result.get("sources"):
        src_tags = " ".join(
            f'<span class="source-tag">📎 {s}</span>'
            for s in sorted(result["sources"])
        )
        st.markdown(f"**Sources:** {src_tags}", unsafe_allow_html=True)

    # Retrieval timing
    ms = result.get("retrieval_ms", 0)
    st.markdown(
        f'<span class="timing-pill">⏱ Retrieval: {ms:.1f} ms</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ──────────────────────────────────────────────
    # Observability Panel (collapsed by default)
    # ──────────────────────────────────────────────
    with st.expander("🔬 Developer Observability Panel", expanded=False):
        chunks = result.get("chunks", [])
        vector_results = result.get("vector_results", [])
        bm25_results = result.get("bm25_results", [])

        # ---- Summary row ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Chunks → LLM", len(chunks))
        c2.metric("Vector Candidates", len(vector_results))
        c3.metric("BM25 Candidates", len(bm25_results))
        c4.metric("Reranker", "ON" if result.get("reranker_enabled") else "OFF")

        st.markdown("---")

        # ---- Score breakdown tabs ----
        tab1, tab2, tab3 = st.tabs(
            ["📊 Final Chunks + Scores", "🔷 Raw Vector Results", "🔶 Raw BM25 Results"]
        )

        def _score_class(v):
            if v is None:
                return ""
            if v >= 0.6:
                return "high"
            if v >= 0.35:
                return "medium"
            return "low"

        def render_chunk_card(chunk: dict, idx: int):
            source = chunk.get("source", "—")
            page = chunk.get("page")
            chunk_idx = chunk.get("chunk_index", "—")
            is_table = chunk.get("is_table", False)
            token_count = chunk.get("token_count", "—")
            doc_type = chunk.get("doc_type", "—")

            vec_score = chunk.get("vector_score")
            bm25_score = chunk.get("bm25_score")
            rrf_score = chunk.get("rrf_score")
            reranker_score = chunk.get("reranker_score")

            page_str = f"p.{page}" if page else "—"

            badges = []
            if vec_score is not None:
                cls = _score_class(vec_score)
                badges.append(f'<span class="score-badge {cls}">vec: {vec_score:.3f}</span>')
            if bm25_score is not None:
                badges.append(f'<span class="score-badge">bm25: {bm25_score:.3f}</span>')
            if rrf_score is not None:
                badges.append(f'<span class="score-badge">rrf: {rrf_score:.5f}</span>')
            if reranker_score is not None:
                badges.append(f'<span class="score-badge">rerank: {reranker_score:.3f}</span>')
            if is_table:
                badges.append('<span class="score-badge table-flag">📊 table</span>')

            text_preview = chunk.get("text", "")
            badge_html = " ".join(badges)

            st.markdown(
                f"""
                <div class="chunk-card">
                  <div class="chunk-meta">
                    <b style="color:#66FCF1;">#{idx}</b>
                    &nbsp;·&nbsp; 📄 {source}
                    &nbsp;·&nbsp; Page {page_str}
                    &nbsp;·&nbsp; chunk_idx={chunk_idx}
                    &nbsp;·&nbsp; {token_count} tokens
                    &nbsp;·&nbsp; .{doc_type}
                    <br>{badge_html}
                  </div>
                  <div class="chunk-text">{text_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab1:
            if not chunks:
                st.caption("No chunks returned.")
            else:
                for i, chunk in enumerate(chunks, start=1):
                    render_chunk_card(chunk, i)

            st.markdown("---")
            st.markdown("**📋 Context block sent to LLM:**")
            ctx = result.get("context_block", "")
            if ctx:
                st.code(ctx, language=None)
            else:
                st.caption("Empty (refusal triggered before context was built).")

        with tab2:
            if not vector_results:
                st.caption("No vector results (no documents indexed?).")
            else:
                for i, chunk in enumerate(vector_results, start=1):
                    render_chunk_card(chunk, i)

        with tab3:
            if not bm25_results:
                st.caption("No BM25 results.")
            else:
                for i, chunk in enumerate(bm25_results, start=1):
                    render_chunk_card(chunk, i)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · HF Embeddings (all-MiniLM-L6-v2) · "
    "BM25 (rank-bm25) · Qdrant · Gemini — **SourceGrounded-RAG**"
)
