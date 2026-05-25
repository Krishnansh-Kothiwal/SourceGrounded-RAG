"""
ingestion/chunker.py — Token-Aware Recursive Chunker

Why recursive chunking beats naive character splitting:
  - Character splitting cuts at fixed byte offsets — mid-sentence, mid-paragraph,
    even mid-word. This destroys semantic coherence and produces chunks that embed
    poorly, since the embedding model receives truncated, incoherent text.
  - Recursive splitting walks down a priority hierarchy of natural text boundaries:
      paragraph break (\n\n) → line break (\n) → sentence end (. ) → word ( ) → char
    It always prefers the coarsest boundary that still fits within the token budget,
    producing chunks aligned to meaning rather than byte position.
  - Token counting via tiktoken ensures chunks fit real LLM context windows precisely.
    The common rule of thumb "1 token ≈ 4 characters" breaks badly for code, tables,
    non-English text, and punctuation-heavy content. tiktoken counts exactly.

Page-aware design:
  - Content is never merged across page boundaries. Each page is chunked independently.
  - If is_table=True, the page is emitted as a single atomic chunk — splitting
    mid-row destroys the tabular structure that makes tables useful.

Config via .env:
  CHUNK_SIZE_TOKENS    — max tokens per chunk (default: 300)
  CHUNK_OVERLAP_TOKENS — token overlap between consecutive chunks (default: 50)
"""

import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# cl100k_base is the GPT-4/text-embedding-3 tokenizer.
# Used here as a model-agnostic token counter — Gemini's actual tokenizer differs
# slightly, but cl100k_base is a reliable conservative proxy.
_ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# Separator hierarchy: coarsest → finest.
# The splitter tries these in order, falling back only when necessary.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_recursive(text: str, separators: list[str], max_tokens: int) -> list[str]:
    """
    Recursively split text using decreasing separator granularity.

    Base case: text fits within max_tokens → return as-is.
    Recursive case: split on the current separator, accumulate segments that
    fit the budget, and recurse on any segment that still exceeds it.
    """
    if _count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    next_separators = separators[1:] if len(separators) > 1 else []

    parts = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + (separator if current else "") + part
        if _count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current.strip():
                if _count_tokens(current) > max_tokens and next_separators:
                    chunks.extend(_split_recursive(current, next_separators, max_tokens))
                else:
                    chunks.append(current)
            current = part

    if current.strip():
        if _count_tokens(current) > max_tokens and next_separators:
            chunks.extend(_split_recursive(current, next_separators, max_tokens))
        else:
            chunks.append(current)

    return chunks


def _add_overlap(segments: list[str], overlap_tokens: int) -> list[str]:
    """
    Prefix each chunk (except the first) with the trailing tokens of the
    previous chunk so that boundary context is not lost.

    Token-counted overlap is more accurate than character-counted overlap
    because it guarantees the prepended prefix stays within the token budget.
    """
    if overlap_tokens == 0 or len(segments) <= 1:
        return segments

    result = [segments[0]]
    for i in range(1, len(segments)):
        prev_tokens = _ENCODING.encode(segments[i - 1])
        overlap_text = _ENCODING.decode(prev_tokens[-overlap_tokens:])
        result.append(overlap_text.strip() + " " + segments[i])
    return result


def chunk_pages(
    pages: list[dict],
    source: str,
    doc_type: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Chunk a preprocessed page list into token-bounded chunks with full metadata.

    Page-aware: each page is chunked independently — content never crosses page
    boundaries unless a single page exceeds chunk_size (forcing a split).
    Tables (is_table=True) are emitted as atomic chunks regardless of size.

    Args:
        pages:        Preprocessed page dicts with 'text', 'page', 'is_table' fields.
        source:       Document filename / display name.
        doc_type:     File extension without dot ('pdf', 'txt', 'md').
        chunk_size:   Max tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks on the same page.

    Returns:
        List of chunk dicts:
        {
            text:        str   — chunk text
            page:        int | None — 1-indexed page number (None for TXT/MD)
            source:      str   — document name
            doc_type:    str   — 'pdf', 'txt', or 'md'
            chunk_index: int   — global sequential index across all chunks
            is_table:    bool  — True if the chunk was flagged as a table
            token_count: int   — actual token count of this chunk
        }
    """
    all_chunks: list[dict] = []
    global_index = 0

    for page in pages:
        text = page["text"]
        page_num = page.get("page")
        is_table = page.get("is_table", False)

        if is_table:
            # Tables must not be split — emit as a single atomic chunk.
            all_chunks.append({
                "text": text.strip(),
                "page": page_num,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": global_index,
                "is_table": True,
                "token_count": _count_tokens(text),
            })
            global_index += 1
            continue

        # Recursively split page text into token-bounded segments.
        segments = _split_recursive(text, _SEPARATORS, chunk_size)

        # Add context overlap between consecutive segments on this page.
        segments = _add_overlap(segments, chunk_overlap)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            all_chunks.append({
                "text": seg,
                "page": page_num,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": global_index,
                "is_table": False,
                "token_count": _count_tokens(seg),
            })
            global_index += 1

    return all_chunks
