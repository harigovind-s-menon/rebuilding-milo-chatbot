# src/reranker/__init__.py
from typing import Any, List

from .dynamic import DynamicReranker, select_best_matches
from .cross_encoder import CrossEncoderReranker

__all__ = ["DynamicReranker", "CrossEncoderReranker", "select_best_matches"]

def get_reranker(name: str, **kwargs):
    """
    Factory to return a reranker instance by name.
    name: "dynamic", "cross_encoder", or "none"
    kwargs are passed to the reranker constructor (e.g., model_name for cross-encoder).
    """
    n = (name or "dynamic").lower()
    if n in ("none", "off", "identity"):
        return DynamicReranker(min_score=0.0, rel_threshold=0.0, gap_threshold=1.0, max_k=kwargs.get("max_k", 8))
    if n in ("dynamic", "threshold"):
        return DynamicReranker(
            min_score=float(kwargs.get("min_score", 0.35)),
            rel_threshold=float(kwargs.get("rel_threshold", 0.72)),
            gap_threshold=float(kwargs.get("gap_threshold", 0.07)),
            max_k=int(kwargs.get("max_k", 8)),
        )
    if n in ("cross", "cross-encoder", "cross_encoder", "crossencoder"):
        model_name = kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoderReranker(model_name=model_name, max_k=int(kwargs.get("max_k", 8)))
    raise ValueError(f"Unknown reranker name: {name}")
