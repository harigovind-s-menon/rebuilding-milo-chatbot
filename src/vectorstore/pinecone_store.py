# src/vectorstore/pinecone_store.py
"""
Wrapper around Pinecone serverless index.
Handles creation, upsert, and querying.
"""

import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")  # default Pinecone serverless env
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rebuilding-milo-index")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # OpenAI text-embedding-3-small uses 1536 dims


def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not found in environment / .env")
    return Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index(pc: Pinecone) -> Any:
    """
    Creates index if it doesn't exist, or returns existing.
    Uses ServerlessSpec which works without choosing regions manually.
    """
    indexes = pc.list_indexes().names()
    if PINECONE_INDEX not in indexes:
        print(f"[pinecone] creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(PINECONE_INDEX)


def upsert_embeddings(
    index,
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    namespace: str = "default"
):
    """
    Upserts a batch of vectors into Pinecone.
    Each metadata dict must include an 'id' key for stable vector IDs.
    """
    # Pinecone expects: {"id": "string", "values": [...], "metadata": {...}}
    vectors = []
    for emb, meta in zip(embeddings, metadatas):
        vec_id = meta.get("id")
        if not vec_id:
            raise ValueError("Each metadata dict must contain an 'id' field")
        vectors.append({"id": vec_id, "values": emb, "metadata": meta})

    index.upsert(vectors=vectors, namespace=namespace)
    print(f"[pinecone] upserted {len(vectors)} vectors to namespace '{namespace}'")


def query_index(
    index,
    embedding: List[float],
    top_k: int = 5,
    namespace: str = "default",
) -> List[Dict[str, Any]]:
    """Query the Pinecone index."""
    res = index.query(
        namespace=namespace,
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return res.matches
