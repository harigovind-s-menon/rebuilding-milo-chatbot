# src/ingestion/pdf_loader.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from pypdf import PdfReader
import re

@dataclass
class BookPage:
    page_number: int
    text: str
    metadata: Dict

def extract_text_by_page(pdf_path: str) -> List[BookPage]:
    """
    Extract text from an editable PDF using pypdf (pure Python).
    Returns a list of BookPage objects with page_number and text.
    """
    reader = PdfReader(pdf_path)
    meta = {}
    try:
        raw_meta = reader.metadata or {}
        # pypdf returns a dictionary-like object for metadata
        meta = {k: raw_meta[k] for k in raw_meta} if raw_meta else {}
    except Exception:
        meta = {}

    pages: List[BookPage] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            # fallback: empty string for that page
            text = ""
        pages.append(BookPage(page_number=i + 1, text=text, metadata=meta))
    return pages

def guess_chapters_from_headings(pages: List[BookPage], heading_pattern: Optional[str] = None) -> List[Dict]:
    """
    Very simple heuristic to detect chapter headings on pages.
    Returns list of dicts with chapter name and start/end pages.
    """
    if heading_pattern is None:
        heading_pattern = r'^(chapter|chapitre|capítulo|capitolo|CHAPTER)\b[\s\w\d\-\:\.]*$'
    chapters = []
    current = None
    for p in pages:
        first_lines = "\n".join(p.text.splitlines()[:5]).strip()
        match = re.search(heading_pattern, first_lines, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            if current:
                current["end_page"] = p.page_number - 1
                chapters.append(current)
            current = {"chapter": first_lines.splitlines()[0].strip(), "start_page": p.page_number, "end_page": None}
    if current:
        current["end_page"] = pages[-1].page_number
        chapters.append(current)
    return chapters