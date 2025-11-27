# src/api/app.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path

import os
from src.embeddings.embedder import embed_texts
from src.vectorstore.pinecone_store import get_pinecone_client, get_or_create_index, query_index

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# add this import near the top of src/api/app.py
from src.llm.prompt import DEFAULT_SYSTEM_PROMPT

# OpenAI responses client
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5")  # change in .env if you have a different name

if OPENAI_API_KEY:
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    _openai_client = None

app = FastAPI(title="Rebuilding Milo — RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the static UI
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# root -> index.html
@app.get("/", include_in_schema=False)
def root_index():
    return FileResponse("src/api/static/index.html")

# where local chunks live (we use the same file the indexer wrote)
DEFAULT_CHUNKS_ROOT = Path("data")

class RagRequest(BaseModel):
    chunks_path: str
    question: str
    top_k: int = 5
    max_context_chars: int = 4000

def load_id_to_text(path: Path) -> Dict[str, Dict[str,Any]]:
    d = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            d[j["id"]] = j
    return d

import json

def make_context_snippets(matches, id2doc, max_chars: int):
    """
    Build a context by concatenating retrieved chunks until max_chars reached.
    Returns the context string and a list of source metadata.
    """
    parts = []
    sources = []
    chars = 0
    for m in matches:
        mid = m.id if hasattr(m, "id") else m["id"]
        meta = m.metadata if hasattr(m, "metadata") else m["metadata"]
        doc = id2doc.get(mid)
        snippet = doc["text"] if doc else ""
        snippet = snippet.replace("\n", " ")
        # shorten chunk to avoid huge context
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " ..."

        if chars + len(snippet) > max_chars and parts:
            break
        parts.append(f"Source (page {meta.get('page_start')}-{meta.get('page_end')}, chunk {meta.get('chunk_index')}):\n{snippet}\n")
        sources.append({"id": mid, "score": m.score if hasattr(m, "score") else m["score"], "meta": meta})
        chars += len(snippet)
    return "\n\n".join(parts), sources

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    # resolve the chunks file
    chunks_path = Path(req.chunks_path)
    if not chunks_path.exists():
        raise HTTPException(status_code=400, detail=f"chunks.jsonl not found: {chunks_path}")

    # load id->doc mapping (local)
    id2doc = {}
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            id2doc[j["id"]] = j

    # 1) embed the question
    try:
        q_emb = embed_texts([req.question], batch_size=1)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

    # 2) query pinecone
    try:
        pc = get_pinecone_client()
        index = get_or_create_index(pc)
        matches = query_index(index, q_emb, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    if not matches:
        return {"answer": "", "sources": [], "reason": "no matches found"}

    # 3) build context
    context, sources = make_context_snippets(matches, id2doc, max_chars=req.max_context_chars)

    # 4) use the external prompt text and build the prompt around it
    system_preamble = DEFAULT_SYSTEM_PROMPT.strip()
    prompt = (
        f"{system_preamble}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {req.question}\n\n"
        f"Provide a concise answer and list which sources you used."
    )

    # 5) call the OpenAI responses API
    if _openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        resp = _openai_client.responses.create(model=LLM_MODEL, input=prompt)

        # Robust extraction of text from the Responses API output:
        text_parts = []

        # 1) Preferred: iterate over resp.output (may be list of objects or dicts)
        for item in getattr(resp, "output", []) or []:
            # item may be an SDK object or a dict
            content = None
            # SDK objects often expose .content
            if hasattr(item, "content"):
                content = item.content
            elif isinstance(item, dict):
                content = item.get("content")

            # content might be a list of pieces (strings or dicts) or a single string
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict):
                        # Many content dicts look like {"type":"output_text","text":"..."} or {"text":"..."}
                        if "text" in c:
                            text_parts.append(c["text"])
                        else:
                            # try common alternatives
                            txt = c.get("string") or c.get("value")
                            if txt:
                                text_parts.append(txt)
            elif isinstance(content, str):
                text_parts.append(content)

        # 2) Fallback — some SDK responses expose output_text
        if not text_parts:
            txt = getattr(resp, "output_text", None)
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt)

        # 3) Final fallback — resp may be dict-like
        if not text_parts and isinstance(resp, dict):
            # try resp.get("output_text") or resp.get("output", [{}])[0].get("content")
            txt = resp.get("output_text") or None
            if txt:
                text_parts.append(txt)
            else:
                out = resp.get("output")
                if out and isinstance(out, list):
                    first = out[0]
                    if isinstance(first, dict):
                        # try common nested shapes
                        c = first.get("content")
                        if isinstance(c, list):
                            # pick text fields from first content item
                            ci = c[0]
                            if isinstance(ci, dict) and "text" in ci:
                                text_parts.append(ci["text"])

        answer_text = "\n\n".join(text_parts).strip() if text_parts else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    return {"answer": answer_text, "sources": sources}
