# src/pipeline/query_pipeline.py
"""
Simple retrieval demo:
  python -m src.pipeline.query_pipeline <chunks.jsonl> "<your question>" [top_k]

Example:
  poetry run python -m src.pipeline.query_pipeline \
    "data/<slug>/chunks.jsonl" "How do I recover from a knee injury?" 5
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import json
from pathlib import Path

from src.embeddings.embedder import embed_texts
from src.vectorstore.pinecone_store import get_pinecone_client, get_or_create_index, query_index

def load_id_to_text(path: Path):
    d = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            d[j["id"]] = j
    return d

def run_query(chunks_jsonl: str, question: str, top_k: int = 5):
    path = Path(chunks_jsonl)
    if not path.exists():
        raise SystemExit(f"Chunks file not found: {path}")
    id2doc = load_id_to_text(path)

    # 1) embed the question
    q_emb = embed_texts([question], batch_size=1)[0]

    # 2) init pinecone and get index
    pc = get_pinecone_client()
    index = get_or_create_index(pc)

    # 3) query
    matches = query_index(index, q_emb, top_k=top_k)

    # 4) print results with local chunk text
    print(f"Top {top_k} results for: {question}\n")
    for i, m in enumerate(matches, start=1):
        mid = m.id if hasattr(m, "id") else m["id"]
        score = m.score if hasattr(m, "score") else m["score"]
        meta = m.metadata if hasattr(m, "metadata") else m["metadata"]
        doc = id2doc.get(mid)
        snippet = (doc["text"][:600].replace("\n", " ")) if doc else "<text not available locally>"
        print(f"--- #{i}  score: {score:.4f}  id: {mid}")
        print(f"metadata: {meta}")
        print(snippet)
        print()
    if not matches:
        print("No matches found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.pipeline.query_pipeline <chunks.jsonl> \"<question>\" [top_k]")
        raise SystemExit(1)
    chunks = sys.argv[1]
    question = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    run_query(chunks, question, top_k=top_k)
