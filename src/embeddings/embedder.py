# src/embeddings/embedder.py
"""
OpenAI v2-compatible embeddings wrapper (openai>=1.0.0 / v2.x).
Uses environment variables:
  - OPENAI_API_KEY
  - EMBED_MODEL (defaults to text-embedding-3-small)
  - BATCH_SIZE
"""
from dotenv import load_dotenv
load_dotenv()
import os
import time
from typing import List, Iterable
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_BATCH = int(os.getenv("BATCH_SIZE", "64"))
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, exponential

_client = None
if OPENAI_API_KEY:
    _client = OpenAI(api_key=OPENAI_API_KEY)

def _chunk_iterable(items: Iterable, n: int):
    it = list(items)
    for i in range(0, len(it), n):
        yield it[i:i + n]

def embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH) -> List[List[float]]:
    """
    Return embeddings for `texts` using OpenAI Python client v1/v2 style (OpenAI()).
    On repeated failure this function will raise an exception instead of returning zero vectors.
    """
    if not texts:
        return []

    if _client is None:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    outs: List[List[float]] = []
    for batch in _chunk_iterable(texts, batch_size):
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
                # resp.data is a list of objects with .embedding (or ['embedding'])
                batch_embs = [d.embedding if hasattr(d, "embedding") else d["embedding"] for d in resp.data]
                outs.extend(batch_embs)
                success = True
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    # fail loudly so we do not insert zero vectors into Pinecone
                    raise RuntimeError(f"[embedder] embedding failed after {MAX_RETRIES} attempts: {e}") from e
                else:
                    wait = RETRY_BACKOFF ** (attempt - 1)
                    print(f"[embedder] embed attempt {attempt} failed: {e}. retrying in {wait}s")
                    time.sleep(wait)
        if not success:
            # should never reach here because we raise on final failure, but keep safe
            raise RuntimeError("[embedder] Unexpected embedding failure")
    return outs
