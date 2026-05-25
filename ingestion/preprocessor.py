"""
ingestion/preprocessor.py — Document Preprocessing

Cleans raw extracted text before chunking:
  - Strip repeated headers/footers across pages (heuristic: line appears on 3+ pages)
  - Normalize whitespace (collapse runs, fix unicode spaces)
  - Detect tables (heuristic: pipe-delimited rows or consistent column spacing)

Input:  list of {"text": str, "page": int | None}
Output: list of {"text": str, "page": int | None, "is_table": bool}
"""

import re
from collections import Counter


def _line_frequency(pages: list[dict]) -> Counter:
    """Count how many distinct pages each non-empty line appears on."""
    freq: Counter = Counter()
    for page in pages:
        # Use a set so a line repeated within one page isn't double-counted
        lines = set(page["text"].splitlines())
        for line in lines:
            stripped = line.strip()
            if stripped:
                freq[stripped] += 1
    return freq


def strip_headers_footers(pages: list[dict]) -> list[dict]:
    """
    Remove lines that appear on ≥3 pages (or ≥50% of pages for short docs).

    Rationale: Page headers ("Company Confidential", page numbers, running titles)
    are the most common source of noise in PDF extraction. They appear verbatim
    on many pages, making them easy to detect without ML.

    Only applied when there are ≥3 pages — fewer pages makes the heuristic
    too aggressive (it would strip legitimately repeated phrases).
    """
    if len(pages) < 3:
        return pages

    freq = _line_frequency(pages)
    threshold = max(3, len(pages) // 2)
    boilerplate = {line for line, count in freq.items() if count >= threshold}

    cleaned = []
    for page in pages:
        lines = page["text"].splitlines()
        filtered = [ln for ln in lines if ln.strip() not in boilerplate]
        cleaned.append({**page, "text": "\n".join(filtered)})
    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace throughout a text block:
      - Replace unicode non-breaking and line-separator spaces
      - Strip trailing whitespace per line
      - Collapse 3+ consecutive blank lines to 2 (preserve intentional paragraph breaks)
    """
    text = text.replace("\xa0", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_table(text: str) -> bool:
    """
    Heuristic table detection. Returns True if the text looks like a table.

    Two signals:
      1. Pipe-delimited: ≥3 lines each containing ≥2 '|' characters
         (catches markdown tables and pipe-extracted PDF tables)
      2. Column-aligned: ≥3 lines each with ≥2 runs of 2+ spaces,
         covering ≥50% of non-empty lines (catches whitespace-aligned tables)

    This is intentionally liberal — false positives produce chunks with
    is_table=True, which just means they won't be split mid-row.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    if pipe_lines >= 3:
        return True

    aligned_lines = sum(1 for ln in lines if len(re.findall(r"  +", ln)) >= 2)
    if aligned_lines >= 3 and aligned_lines / len(lines) >= 0.5:
        return True

    return False


def preprocess_pages(pages: list[dict]) -> list[dict]:
    """
    Full preprocessing pipeline for a document's page list.

    Steps:
      1. Strip headers/footers (cross-page boilerplate removal)
      2. Normalize whitespace per page
      3. Flag table-like pages with is_table=True

    Args:
        pages: Raw page list from a document loader.

    Returns:
        Cleaned page list with 'is_table' field added. Empty pages dropped.
    """
    pages = strip_headers_footers(pages)
    result = []
    for page in pages:
        cleaned = normalize_whitespace(page["text"])
        if cleaned:
            result.append({
                "text": cleaned,
                "page": page["page"],
                "is_table": _is_table(cleaned),
            })
    return result
