import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Key Checks (before any imports that use them) ─────────────
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
from rag_pipeline import ingest_document, ask
from retrieval.reranker import ENABLE_RERANKER

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SourceGrounded RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

/* ── Root & body ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stMain"] {
    background: #0A0A0A !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #1E1E1E !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

/* ── Typography overrides ── */
h1, h2, h3, h4, p, label, span, div {
    font-family: 'Syne', sans-serif;
}

/* ── Streamlit default element cleanup ── */
[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #D4A853 !important;
}
[data-testid="stSlider"] label {
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111 !important;
    border: 1px dashed #222 !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"] label {
    color: #EEEEEE !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 18px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #111 !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #EEEEEE !important;
    font-family: 'DM Mono', monospace !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: background 0.15s !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #D4A853 !important;
}

/* ── Text area ── */
[data-testid="stTextArea"] textarea {
    background: #111 !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 16px !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #D4A853 !important;
    box-shadow: none !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #EEEEEE !important;
}
[data-testid="stTextArea"] label {
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: #D4A853 !important;
    color: #0A0A0A !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: opacity 0.15s !important;
    padding: 0.5rem 1.25rem !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
    background: #D4A853 !important;
    color: #0A0A0A !important;
}
[data-testid="stButton"] button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
}

/* ── Divider ── */
hr {
    border-color: #1A1A1A !important;
    margin: 0 !important;
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: #111 !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 18px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #D4A853 !important;
}

/* ── Markdown ── */
[data-testid="stMarkdown"] {
    color: #FFFFFF !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111 !important;
    border: 1px solid #1E1E1E !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #FFFFFF; }

</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def mono(text: str, color: str = "#EEEEEE") -> str:
    return f'<span style="font-family:\'DM Mono\',monospace;font-size:22px;color:{color};letter-spacing:0.1em;text-transform:uppercase;">{text}</span>'

def section_header(num: str, title: str, meta: str = "") -> None:
    right = f'<span style="font-family:\'DM Mono\',monospace;font-size:22px;color:#FFFFFF;letter-spacing:0.05em;">{meta}</span>' if meta else ""
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:0.75rem;">
        <div>
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:#FFFFFF;letter-spacing:0.12em;text-transform:uppercase;">{num} —</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;letter-spacing:-0.02em;color:#FFFFFF;">{title}</div>
        </div>
        {right}
    </div>
    """, unsafe_allow_html=True)


def status_bar(docs_loaded: bool, reranker: bool, query_ready: bool) -> None:
    def chip(label, on):
        dot_color = "#D4A853" if on else "#EEEEEE"
        text_color = "#888" if on else "#FFFFFF"
        return (f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="width:5px;height:5px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;color:{text_color};letter-spacing:0.08em;text-transform:uppercase;">{label}</span>'
                f'</div>')

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1.5rem;margin-top:0.75rem;
                padding-top:0.75rem;border-top:1px solid #1A1A1A;">
        {chip("Docs indexed" if docs_loaded else "No docs indexed", docs_loaded)}
        {chip("Reranker on", reranker)}
        {chip("Query ready" if query_ready else "Awaiting query", query_ready)}
    </div>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = []
if "answer" not in st.session_state:
    st.session_state.answer = None
if "sources" not in st.session_state:
    st.session_state.sources = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:22px;color:#EEEEEE;
                    letter-spacing:0.15em;text-transform:uppercase;">System</div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    letter-spacing:-0.03em;color:#FFFFFF;margin-top:2px;">SourceGrounded</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #1E1E1E;padding-top:1rem;margin-bottom:0.5rem;">', unsafe_allow_html=True)
    st.markdown(mono("Document library"), unsafe_allow_html=True)

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:22px;color:#FFFFFF;
                        border:1px solid #1E1E1E;padding:5px 8px;border-radius:2px;
                        margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {doc["name"]} ({doc["chunks"]} chunks)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:18px;color:#EEEEEE;
                    border:1px dashed #1E1E1E;padding:6px 10px;border-radius:2px;margin-top:6px;">
            — empty —
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #1E1E1E;padding-top:1rem;margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown(mono("Retrieval"), unsafe_allow_html=True)
    top_k = st.slider("Candidates per retriever (top-k)", min_value=1, max_value=50, value=int(os.getenv("RETRIEVAL_TOP_K", "10")), label_visibility="collapsed")
    st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:13px;color:#FFFFFF;margin-top:2px;">top-k = {top_k}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    enable_reranker = ENABLE_RERANKER
    st.markdown(f"""
    <div style="border-top:1px solid #1E1E1E;padding-top:1rem;margin-top:1rem;">
        <div style="font-family:'DM Mono',monospace;font-size:13px;color:#EEEEEE;
                    letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.5rem;">Pipeline</div>
        <div style="display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;
                    font-size:22px;color:#EEEEEE;">
            <div style="width:6px;height:6px;border-radius:50%;background:{'#4CAF50' if enable_reranker else '#F44336'};flex-shrink:0;"></div>
            Reranker {'active' if enable_reranker else 'disabled'}
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:13px;color:#FFFFFF;
                    margin-top:4px;letter-spacing:0.05em;">Vector + BM25 → RRF</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

# Topbar
st.markdown("""
<div style="display:flex;align-items:baseline;gap:1.5rem;
            padding-bottom:1.25rem;border-bottom:1px solid #1A1A1A;margin-bottom:1.5rem;">
    <div style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;
                letter-spacing:-0.04em;color:#FFFFFF;">RAG</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span style="font-family:'DM Mono',monospace;font-size:13px;letter-spacing:0.1em;
                     text-transform:uppercase;color:#EEEEEE;border:1px solid #222;
                     padding:2px 8px;border-radius:2px;">Hybrid retrieval</span>
        <span style="font-family:'DM Mono',monospace;font-size:13px;letter-spacing:0.1em;
                     text-transform:uppercase;color:#EEEEEE;border:1px solid #222;
                     padding:2px 8px;border-radius:2px;">Source-grounded</span>
        <span style="font-family:'DM Mono',monospace;font-size:13px;letter-spacing:0.1em;
                     text-transform:uppercase;color:#EEEEEE;border:1px solid #222;
                     padding:2px 8px;border-radius:2px;">Multi-doc</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── 01 Upload ─────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div style="border:1px solid #1E1E1E;border-radius:4px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">', unsafe_allow_html=True)
    section_header("01", "Upload documents", "PDF · TXT · MD")

    uploaded_files = st.file_uploader(
        "drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Process & Index", disabled=not uploaded_files):
        with st.spinner("Indexing documents..."):
            total = len(uploaded_files)
            new_docs = []
            
            for i, uploaded_file in enumerate(uploaded_files):
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
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)
            
            if new_docs:
                st.session_state.documents = new_docs
                st.rerun()

    if uploaded_files:
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:22px;color:#D4A853;
                    margin-top:0.5rem;letter-spacing:0.05em;">
            {len(uploaded_files)} file(s) selected
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── 02 Query ──────────────────────────────────────────────────────────────────
docs_loaded = bool(st.session_state.documents)

with st.container():
    st.markdown('<div style="border:1px solid #1E1E1E;border-radius:4px;padding:1.25rem 1.5rem;">', unsafe_allow_html=True)
    section_header("02", "Ask a question")

    query = st.text_area(
        "query",
        placeholder="Upload a document first..." if not docs_loaded else "What does the document say about...",
        height=96,
        disabled=not docs_loaded,
        label_visibility="collapsed",
    )

    col_run, col_clear = st.columns([1, 5])
    with col_run:
        run = st.button("Run ↗", disabled=not (docs_loaded and query))
    with col_clear:
        if st.button("Clear", disabled=not st.session_state.answer) if st.session_state.answer else False:
            st.session_state.answer = None
            st.session_state.sources = []
            st.rerun()

    status_bar(docs_loaded, enable_reranker, bool(docs_loaded and query))
    st.markdown('</div>', unsafe_allow_html=True)


# ── Answer area ───────────────────────────────────────────────────────────────
if run and query:
    with st.spinner("Generating answer..."):
        try:
            result = ask(query.strip(), top_k=top_k)
            
            if result.get("refused"):
                st.session_state.answer = f"⚠️ {result['answer']}\n\n**Reason:** {result.get('refusal_reason', '')}"
            else:
                st.session_state.answer = result["answer"]
                
            st.session_state.sources = [
                {
                    "source": c.get("source", "Unknown"), 
                    "page": c.get("page") or "N/A", 
                    "excerpt": c.get("text", "")
                } 
                for c in result.get("chunks", [])
            ]
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.session_state.answer = None

if st.session_state.answer:
    st.markdown("""
    <div style="border-top:1px solid #1A1A1A;margin-top:1.5rem;padding-top:1.5rem;">
        <div style="font-family:'DM Mono',monospace;font-size:13px;color:#FFFFFF;
                    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.75rem;">Answer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:22px;color:#FFFFFF;
                line-height:1.7;padding:1rem;background:#111;border:1px solid #1E1E1E;border-radius:2px;">
        {st.session_state.answer}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.sources:
        st.markdown("""
        <div style="margin-top:1rem;">
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:#FFFFFF;
                        letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">Sources</div>
        </div>
        """, unsafe_allow_html=True)
        for s in st.session_state.sources:
            with st.expander(f"{s['source']} — p.{s['page']}"):
                st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:18px;color:#FFFFFF;">{s["excerpt"]}</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:0.75rem;border-top:1px solid #111;
            display:flex;gap:1rem;align-items:center;flex-wrap:wrap;">
    <span style="font-family:'DM Mono',monospace;font-size:13px;color:#EEEEEE;letter-spacing:0.08em;text-transform:uppercase;">Streamlit</span>
    <span style="color:#1E1E1E;">·</span>
    <span style="font-family:'DM Mono',monospace;font-size:13px;color:#EEEEEE;letter-spacing:0.08em;text-transform:uppercase;">all-MiniLM-L6-v2</span>
    <span style="color:#1E1E1E;">·</span>
    <span style="font-family:'DM Mono',monospace;font-size:13px;color:#EEEEEE;letter-spacing:0.08em;text-transform:uppercase;">Qdrant</span>
    <span style="color:#1E1E1E;">·</span>
    <span style="font-family:'DM Mono',monospace;font-size:13px;color:#EEEEEE;letter-spacing:0.08em;text-transform:uppercase;">Gemini</span>
</div>
""", unsafe_allow_html=True)
