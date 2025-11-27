# src/ingestion/text_cleaner.py
import re

def clean_text(text: str) -> str:
    """
    Lightweight cleaning suited for RAG ingestion:
    - Remove control characters
    - Fix hyphenation broken at line breaks (e.g., "exam-\nple" -> "example")
    - Normalize whitespace and paragraph breaks
    """
    if not text:
        return ""
    # remove control chars except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # fix hyphenation broken at end-of-line: word-\nword -> wordword
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # replace newlines that are mid-sentence with space (naive heuristic)
    text = re.sub(r'(?<=[^\.\!\?\n])\n(?=[^\nA-Z0-9])', ' ', text)
    # collapse multiple newlines to paragraph breaks
    text = re.sub(r'\n{2,}', '\n\n', text)
    # collapse extra whitespace and tabs
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()