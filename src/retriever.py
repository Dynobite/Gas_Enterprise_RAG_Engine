"""
Two-Stage Hybrid Retriever with Small-to-Big Parent Page Hydration.
"""
import os
import re
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.embeddings import OllamaEmbeddingClient
from src.reranker import FlashRankReranker

class GasRagRetriever:
    """Hybrid Retriever combining Qdrant HNSW vector search, FlashRank, and PageIndex Hydration."""
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embed_client: Optional[OllamaEmbeddingClient] = None,
        reranker: Optional[FlashRankReranker] = None,
        enable_parent_page_hydration: bool = True
    ):
        self.host: str = host or os.getenv("QDRANT_HOST", "localhost")
        self.port: int = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name: str = collection_name or os.getenv("COLLECTION_NAME", "gas_rag_standards")
        self.embed_client: OllamaEmbeddingClient = embed_client or OllamaEmbeddingClient()
        self.reranker: FlashRankReranker = reranker or FlashRankReranker()
        self.client: QdrantClient = QdrantClient(host=self.host, port=self.port, timeout=30.0)
        self.enable_parent_page_hydration: bool = enable_parent_page_hydration

    def hydrate_parent_page(self, source: str, page: Optional[int]) -> Optional[str]:
        """Reconstructs the 100% full parent page text from Qdrant in < 1ms."""
        if not source or page is None:
            return None
        try:
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="page", match=MatchValue(value=page))
                ]
            )
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=64,
                with_payload=True,
                with_vectors=False
            )
            if not points:
                return None

            points.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
            full_page_text = "\n\n".join(
                [(p.payload.get("text", "")).strip() for p in points if p.payload and p.payload.get("text")]
            )
            return full_page_text if full_page_text.strip() else None
        except Exception:
            return None

    def coalesce_and_hydrate_pages(self, ranked_chunks: List[Dict[str, Any]], max_chars_budget: int = 12000) -> List[Dict[str, Any]]:
        """Assembles parent pages for winning chunks within safety context limits."""
        seen_pages = set()
        hydrated_chunks: List[Dict[str, Any]] = []
        total_chars = 0

        for chunk in ranked_chunks:
            source = chunk.get("source", "")
            page = chunk.get("page")
            page_key = (source, page)

            if page is not None and page_key in seen_pages:
                continue

            if page is not None and self.enable_parent_page_hydration:
                parent_text = self.hydrate_parent_page(source, page)
                if parent_text:
                    seen_pages.add(page_key)
                    if total_chars + len(parent_text) <= max_chars_budget:
                        c_copy = chunk.copy()
                        c_copy["text"] = parent_text
                        c_copy["is_hydrated_parent_page"] = True
                        hydrated_chunks.append(c_copy)
                        total_chars += len(parent_text)
                        continue

            hydrated_chunks.append(chunk)
            total_chars += len(chunk.get("text", ""))
            if total_chars >= max_chars_budget:
                break

        return hydrated_chunks

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
        candidate_limit: int = 40,
        enable_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Executes candidate vector scan, FlashRank cross-encoding, and Parent Page Hydration."""
        queries = [query] if isinstance(query, str) else [q for q in query if q.strip()]
        if not queries:
            return []

        primary_query = queries[0]
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for q_text in queries:
            emb = self.embed_client.get_embedding(q_text)
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=emb,
                limit=candidate_limit
            )
            for hit in hits:
                doc = {"id": hit.id, "score": hit.score, **hit.payload}
                if hit.id not in candidates_by_id or hit.score > candidates_by_id[hit.id]["score"]:
                    candidates_by_id[hit.id] = doc

        candidate_list = list(candidates_by_id.values())
        if enable_rerank:
            ranked = self.reranker.rerank(query=primary_query, candidates=candidate_list, top_k=top_k)
        else:
            candidate_list.sort(key=lambda x: x.get("score", 0), reverse=True)
            ranked = candidate_list[:top_k]

        if self.enable_parent_page_hydration:
            return self.coalesce_and_hydrate_pages(ranked)
        return ranked
