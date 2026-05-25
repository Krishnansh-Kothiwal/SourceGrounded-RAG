"""
ingestion/loaders.py — Document Loaders

Dispatches file loading by extension. Each loader returns a list of page dicts:
    {"text": str, "page": int | None}

Page numbers are 1-indexed where available (PDFs), None for flat files (TXT, MD).
Architecture is extensible: add new loaders to _LOADERS to support more formats.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:
        List of {"text": str, "page": int} dicts, one per non-empty page.
        Page numbers are 1-indexed.
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page": i})
    return pages


def load_txt(file_path: str) -> list[dict]:
    """Load a plain-text file as a single logical 'page'."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [{"text": text, "page": None}]


def load_markdown(file_path: str) -> list[dict]:
    """
    Load a Markdown file as a single logical 'page'.
    Raw Markdown text is preserved (not rendered to HTML) so the chunker
    can split on paragraph breaks naturally.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [{"text": text, "page": None}]


# Registry: extension → loader function.
# Add entries here to support additional file types without changing load_document().
_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_markdown,
}


def load_document(file_path: str) -> list[dict]:
    """
    Route a file to the correct loader by extension.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        List of page dicts: [{"text": str, "page": int | None}, ...]

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        supported = ", ".join(_LOADERS.keys())
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )
    return loader(file_path)


def supported_extensions() -> list[str]:
    """Return list of supported file extensions (e.g. ['.pdf', '.txt', '.md'])."""
    return list(_LOADERS.keys())
