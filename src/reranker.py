"""
Cross-Encoder Reranker Module.
Performs second-stage deep cross-attention re-scoring using FlashRank.
"""
from typing import List, Dict, Any, Optional

try:
    from flashrank import Ranker, RerankRequest
    HAS_FLASHRANK = True
except ImportError:
    HAS_FLASHRANK = False


class FlashRankReranker:
    """Second-stage neural reranker using FlashRank (ms-marco-MiniLM-L-12-v2)."""
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        if HAS_FLASHRANK:
            self.ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        else:
            self.ranker = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        dense_weight: float = 0.45,
        rerank_weight: float = 0.55
    ) -> List[Dict[str, Any]]:
        """Re-scores candidate chunks with blended dense + cross-encoder scoring."""
        if not candidates:
            return []

        if not HAS_FLASHRANK or self.ranker is None:
            return candidates[:top_k]

        passages = [
            {"id": i, "text": c.get("text", ""), "meta": c}
            for i, c in enumerate(candidates)
        ]

        try:
            req = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(req)

            scored = []
            for r in results:
                cand = r["meta"]
                dense_score = float(cand.get("score", 0.0))
                cross_score = float(r.get("score", 0.0))
                blended = (dense_weight * dense_score) + (rerank_weight * cross_score)
                cand["rerank_score"] = round(cross_score, 4)
                cand["final_score"] = round(blended, 4)
                scored.append(cand)

            scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return scored[:top_k]
        except Exception:
            return candidates[:top_k]
