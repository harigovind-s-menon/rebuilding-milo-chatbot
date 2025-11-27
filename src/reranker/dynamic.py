# src/reranker/dynamic.py
from typing import List, Any, Callable

def _score_of(m: Any) -> float:
    if hasattr(m, "score"):
        return float(m.score or 0.0)
    if isinstance(m, dict):
        return float(m.get("score", 0.0) or 0.0)
    # fallback
    return 0.0

def select_best_matches(matches: List[Any],
                        min_score: float = 0.35,
                        rel_threshold: float = 0.72,
                        gap_threshold: float = 0.07,
                        max_k: int = 10) -> List[Any]:
    if not matches:
        return []
    selected = []
    top_score = _score_of(matches[0])
    prev_score = top_score
    for i, m in enumerate(matches):
        s = _score_of(m)
        if s < min_score:
            break
        if s < top_score * rel_threshold:
            break
        if (prev_score - s) > gap_threshold and i > 0:
            break
        selected.append(m)
        prev_score = s
        if len(selected) >= max_k:
            break
    return selected

class DynamicReranker:
    def __init__(self, min_score: float = 0.35, rel_threshold: float = 0.72, gap_threshold: float = 0.07, max_k: int = 10):
        self.min_score = min_score
        self.rel_threshold = rel_threshold
        self.gap_threshold = gap_threshold
        self.max_k = max_k

    def rerank(self, query: str, matches: List[Any]) -> List[Any]:
        # matches are expected sorted best->worst already
        return select_best_matches(matches, min_score=self.min_score, rel_threshold=self.rel_threshold, gap_threshold=self.gap_threshold, max_k=self.max_k)
