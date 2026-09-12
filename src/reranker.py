"""
Cross-Encoder Reranker Module for GASlight-Me RAG.
Performs deep query-document cross-attention re-scoring on top candidate chunks
to maximize Precision@K and filter out semantic false positives.
"""

from typing import List, Dict, Any, Optional

try:
    from flashrank import Ranker as _FlashRankRanker, RerankRequest as _RerankRequest
    _HAS_FLASHRANK = True
except ImportError:
    _HAS_FLASHRANK = False
    _FlashRankRanker = None  # type: ignore[assignment,misc]
    _RerankRequest = None  # type: ignore[assignment,misc]


class GasRagReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled and _HAS_FLASHRANK
        self._ranker: Optional[object] = None
        if not _HAS_FLASHRANK and enabled:
            print("[WARN] flashrank is not installed. Reranker disabled. Run: pip install flashrank")

    def _get_ranker(self) -> Optional[object]:
        if self._ranker is None and _HAS_FLASHRANK:
            try:
                self._ranker = _FlashRankRanker(model_name=self.model_name)
            except Exception as exc:
                print(f"[WARN] Failed to initialize FlashRank ({exc}). Reranker disabled.")
                self.enabled = False
        return self._ranker

    def rerank(
        self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder scoring."""
        if not self.enabled or not chunks:
            return chunks[:top_n]

        ranker = self._get_ranker()
        if ranker is None:
            return chunks[:top_n]

        try:
            passages = [
                {"id": idx, "text": c.get("text", "")}
                for idx, c in enumerate(chunks)
            ]
            rerank_req = _RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(rerank_req)

            # Calculate blended hybrid score (Multilingual BGE-M3 Dense + Cross-Encoder Attention + Exact Source Match)
            rescored_chunks: List[Dict[str, Any]] = []
            q_words = [w.strip(",.()[]{}:;\"'").lower() for w in query.split() if len(w) >= 4]

            for r in ranked_results:
                orig_idx = r["id"]
                chunk_copy = dict(chunks[orig_idx])
                reranker_score = float(r.get("score", 0))
                vec_score = float(chunk_copy.get("score", 0))

                src = str(chunk_copy.get("source", "")).lower()
                exact_doc_match = any(w in src for w in q_words)
                doc_bonus = 0.40 if exact_doc_match else 0.0

                blended = (0.45 * vec_score) + (0.55 * reranker_score) + doc_bonus

                chunk_copy["reranker_score"] = round(reranker_score, 4)
                chunk_copy["initial_vector_score"] = round(vec_score, 4)
                chunk_copy["score"] = round(blended, 4)
                rescored_chunks.append(chunk_copy)

            rescored_chunks.sort(key=lambda x: x["score"], reverse=True)
            return rescored_chunks[:top_n]

        except Exception as exc:
            print(f"[WARN] Error during reranking: {exc}. Falling back to vector ranking.")
            return chunks[:top_n]
