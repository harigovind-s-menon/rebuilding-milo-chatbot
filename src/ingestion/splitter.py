# src/ingestion/splitter.py
from typing import List, Iterable, Dict
from .tokenizer import Tokenizer
import uuid

def chunk_pages(
    pages: List[Dict],
    tokenizer: Tokenizer,
    max_tokens: int = 800,
    overlap_tokens: int = 128
) -> Iterable[Dict]:
    """
    Token-aware chunking over pages. Yields chunk dicts with:
      - id
      - text
      - token_count
      - page_start
      - page_end
      - metadata

    Strategy:
      - Build a buffer by paragraph.
      - If buffer + para <= max_tokens: append.
      - Else: flush buffer as a chunk and start a new buffer.
      - If a single paragraph > max_tokens: split by sentence-like boundaries.
      - The overlap parameter is accepted for API compatibility but overlap is implemented
        naively by carrying the last partial buffer into the next chunk start when needed.
    """
    buffer_text = ""
    buffer_start_page = None
    buffer_end_page = None

    for p in pages:
        text = p["text"]
        page_num = p["page_number"]
        paras = [para.strip() for para in text.split("\n\n") if para.strip()]

        for para in paras:
            if buffer_text == "":
                buffer_start_page = page_num

            current_tokens = tokenizer.count_tokens(buffer_text) if buffer_text else 0
            para_tokens = tokenizer.count_tokens(para)

            if current_tokens + para_tokens <= max_tokens:
                # safe to append paragraph to buffer
                buffer_text = (buffer_text + "\n\n" + para) if buffer_text else para
                buffer_end_page = page_num
            else:
                # flush existing buffer as a chunk
                if buffer_text:
                    yield {
                        "id": str(uuid.uuid4()),
                        "text": buffer_text.strip(),
                        "token_count": tokenizer.count_tokens(buffer_text),
                        "page_start": buffer_start_page,
                        "page_end": buffer_end_page,
                        "metadata": p.get("metadata", {}),
                    }

                    # create overlap by keeping last 'overlap_tokens' worth of text if possible
                    # naive approach: keep last N words approximating overlap_tokens
                    if overlap_tokens > 0:
                        words = buffer_text.split()
                        keep_words = max(0, min(len(words), int(overlap_tokens * 1.5)))  # heuristic
                        overlap_text = " ".join(words[-keep_words:]) if keep_words > 0 else ""
                    else:
                        overlap_text = ""

                    # start new buffer with overlap + current paragraph (or just paragraph)
                    buffer_text = (overlap_text + "\n\n" + para).strip() if overlap_text else para
                    buffer_start_page = page_num
                    buffer_end_page = page_num
                else:
                    # paragraph itself larger than max_tokens -> split by sentences (naive)
                    sentences = [s.strip() for s in para.split(". ") if s.strip()]
                    temp = ""
                    temp_start = page_num
                    for s in sentences:
                        s = s.strip()
                        if not s:
                            continue
                        s_tokens = tokenizer.count_tokens(s)
                        if tokenizer.count_tokens(temp) + s_tokens <= max_tokens:
                            temp = (temp + ". " + s).strip() if temp else s
                        else:
                            yield {
                                "id": str(uuid.uuid4()),
                                "text": temp.strip(),
                                "token_count": tokenizer.count_tokens(temp),
                                "page_start": temp_start,
                                "page_end": page_num,
                                "metadata": p.get("metadata", {}),
                            }
                            temp = s
                            temp_start = page_num
                    if temp:
                        buffer_text = temp
                        buffer_start_page = temp_start
                        buffer_end_page = page_num

    # flush remaining buffer
    if buffer_text:
        yield {
            "id": str(uuid.uuid4()),
            "text": buffer_text.strip(),
            "token_count": tokenizer.count_tokens(buffer_text),
            "page_start": buffer_start_page,
            "page_end": buffer_end_page,
            "metadata": pages[-1].get("metadata", {}) if pages else {},
        }
