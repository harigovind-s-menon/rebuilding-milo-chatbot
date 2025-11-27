# src/pipeline/index_pipeline.py
"""
Read chunks.jsonl -> embed -> upsert to Pinecone.
Usage:
  poetry run python -m src.pipeline.index_pipeline data/<slug>/chunks.jsonl
"""

# load .env if present
from dotenv import load_dotenv
load_dotenv()
import os
import json
from pathlib import Path
from typing import List, Dict

from src.embeddings.embedder import embed_texts
from src.vectorstore.pinecone_store import get_pinecone_client, get_or_create_index, upsert_embeddings

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

def load_chunks(path: Path) -> List[Dict]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def run_index(chunks_jsonl: str, namespace: str = NAMESPACE, batch_size: int = BATCH_SIZE):
    p = Path(chunks_jsonl)
    if not p.exists():
        raise SystemExit(f"Missing chunks file: {p}")

    docs = load_chunks(p)
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = []
    for d in docs:
        meta = {
            "id": d["id"],
            "book_title": d.get("book_title"),
            "book_slug": d.get("book_slug"),
            "chunk_index": d.get("chunk_index"),
            "page_start": d.get("page_start"),
            "page_end": d.get("page_end"),
            "source": d.get("source"),
        }
        metas.append(meta)

    print(f"[indexer] embedding {len(texts)} chunks (batch_size={batch_size})...")
    # produce embeddings in batches
    embeddings = embed_texts(texts, batch_size=batch_size)
    if not embeddings:
        raise SystemExit("[indexer] no embeddings produced; check OPENAI_API_KEY and network")

    # init pinecone and index
    pc = get_pinecone_client()
    index = get_or_create_index(pc)

    # upsert in batches
    total = len(embeddings)
    for i in range(0, total, batch_size):
        emb_batch = embeddings[i:i+batch_size]
        meta_batch = metas[i:i+batch_size]
        print(f"[indexer] upserting batch {i}-{i+len(emb_batch)-1} ({len(emb_batch)} vectors)...")
        upsert_embeddings(index, emb_batch, meta_batch, namespace=namespace)

    print(f"[indexer] Done. Upserted {total} vectors into Pinecone index '{os.getenv('PINECONE_INDEX', 'unknown')}' (namespace='{namespace}').")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.index_pipeline <chunks.jsonl>")
        raise SystemExit(1)
    run_index(sys.argv[1])
