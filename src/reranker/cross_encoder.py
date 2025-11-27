# src/reranker/cross_encoder.py
from typing import List, Any
import math

class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper using sentence-transformers' CrossEncoder.
    This class defers importing heavy deps until used.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_k: int = 8, batch_size: int = 32):
        self.model_name = model_name
        self.max_k = max_k
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RuntimeError("CrossEncoder not available. Install sentence-transformers and torch (e.g. `poetry add sentence-transformers torch`).") from e
        self._model = CrossEncoder(self.model_name)

    def _score_pairs(self, query: str, texts: List[str]) -> List[float]:
        self._ensure_model()
        # CrossEncoder accepts list of (query, passage) pairs
        pairs = [(query, t) for t in texts]
        # batch inference
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]
            batch_scores = self._model.predict(batch, show_progress_bar=False)
            # ensure list of floats
            batch_scores = [float(s) for s in batch_scores]
            scores.extend(batch_scores)
        return scores

    def rerank(self, query: str, matches: List[Any]) -> List[Any]:
        """
        Accepts Pinecone matches (list of SDK objects or dicts). Returns matches re-ordered by cross-encoder score (descending),
        and truncates to self.max_k.
        """
        if not matches:
            return []

        # extract texts and keep mapping to original match
        texts = []
        for m in matches:
            # metadata or local doc text might be available; first try metadata->we expect 'metadata' to exist
            meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else None)
            text = None
            if meta and isinstance(meta, dict) and "text" in meta:
                text = meta.get("text")
            # fallback to 'id' and rely on caller to fetch full text (we will use metadata.source or id)
            if text is None:
                # try to use a short passage if available in match (some pinecone clients include snippet)
                text = getattr(m, "metadata", None) and getattr(m.metadata, "snippet", None) or None
            if text is None:
                # as last resort, store an empty placeholder (cross-encoder will give low scores)
                text = ""
            texts.append(text)

        # compute cross-encoder scores; if all texts empty, return top-k of original
        if all(not t for t in texts):
            return matches[:self.max_k]

        try:
            scores = self._score_pairs(query, texts)
        except Exception as e:
            # if model fails, fallback to returning original topk
            return matches[:self.max_k]

        # attach scores and sort
        scored = list(zip(matches, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [m for m, s in scored][:self.max_k]
        return reranked
