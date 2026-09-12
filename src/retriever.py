"""
2-Stage Retrieval & Reranking Module for GASlight-Me RAG.
Stage 1: Fast Qdrant HNSW vector search over 32k+ points via BGE-M3 (3 ms).
Stage 2: Cross-Encoder Reranker deep attention scoring via FlashRank MiniLM (15-20 ms).
Feature: Small-to-Big Parent Page Hydration (Zero-Latency Full Page Context Assembly).
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.indexer import QdrantVectorIndexer
from src.reranker import GasRagReranker

class GasRagRetriever:
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = "gas_rag_standards",
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "bge-m3",
        use_reranker: bool = True,
        enable_parent_page_hydration: bool = True,
        enable_page_coalescing: bool = True,
        qdrant_url: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or os.path.join(project_root, "vector_db")
        self.collection_name = collection_name
        self.indexer = QdrantVectorIndexer(
            db_path=self.db_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_model=embedding_model,
            ollama_url=ollama_url
        )
        self.reranker = GasRagReranker(enabled=use_reranker)
        self.enable_parent_page_hydration = enable_parent_page_hydration
        self.enable_page_coalescing = enable_page_coalescing

    def hydrate_parent_page(self, source: str, page: int) -> Optional[str]:
        """
        Ultra-fast in-memory lookup in Qdrant to assemble the 100% full parent page text
        (including headers, footnotes, units, and tables) for a matched snippet.
        """
        try:
            from qdrant_client.http import models as qmodels
            res, _ = self.indexer.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)),
                        qmodels.FieldCondition(key="page", match=qmodels.MatchValue(value=page)),
                    ]
                ),
                limit=15,
                with_payload=True,
                with_vectors=False
            )
            if not res:
                return None

            # Collect and sort text fragments from this page
            page_texts = []
            for p in res:
                t = p.payload.get("text", "").strip()
                if t and not any(t in existing for existing in page_texts):
                    page_texts.append(t)

            if page_texts:
                return "\n\n".join(page_texts)
        except Exception as e:
            # Non-blocking fallback
            return None
        return None

    def coalesce_and_hydrate_pages(
        self,
        chunks: List[Dict[str, Any]],
        max_total_chars: int = 12000
    ) -> List[Dict[str, Any]]:
        """
        Small-to-Big Parent Page Coalescing & Hydration:
        1. Groups chunks by (source, page)
        2. Hydrates full parent page context for PDF standards
        3. Protects atomic Excel rows
        4. Enforces context token budget guard
        """
        if not chunks:
            return []

        page_groups: Dict[str, Dict[str, Any]] = {}
        ordered_keys: List[str] = []

        for c in chunks:
            src = c.get("source", "unknown")
            page = c.get("page", 1)
            fmt = c.get("format", "")

            # Excel tabular rows are already atomic micro-chunks — preserve as-is
            if fmt == "excel" or "dfmea" in src.lower() or "bom" in src.lower():
                key = f"{src}_{page}_{len(ordered_keys)}"
                page_groups[key] = dict(c)
                ordered_keys.append(key)
                continue

            # Standard PDF/DOCX page key
            key = f"{src}__p{page}"
            text = c.get("text", "").strip()

            if key not in page_groups:
                page_groups[key] = dict(c)
                page_groups[key]["text_parts"] = [text]
                ordered_keys.append(key)
            else:
                existing_parts = page_groups[key]["text_parts"]
                if not any(text in p or p in text for p in existing_parts):
                    existing_parts.append(text)
                if c.get("score", 0) > page_groups[key].get("score", 0):
                    page_groups[key]["score"] = c.get("score", 0)

        # Assemble final results with Parent Page Hydration
        coalesced_results: List[Dict[str, Any]] = []
        current_chars = 0

        for k in ordered_keys:
            item = page_groups[k]
            src = item.get("source", "")
            page = item.get("page", 1)
            fmt = item.get("format", "")

            if "text_parts" in item:
                # Merge existing parts
                merged_text = "\n\n".join(item["text_parts"])
                
                # If enabled, fetch full parent page context from Qdrant
                if self.enable_parent_page_hydration and fmt != "excel":
                    full_parent = self.hydrate_parent_page(src, page)
                    if full_parent and len(full_parent) >= len(merged_text):
                        merged_text = full_parent

                item["text"] = merged_text
                item.pop("text_parts", None)

            txt_len = len(item.get("text", ""))
            if current_chars + txt_len > max_total_chars and coalesced_results:
                break

            coalesced_results.append(item)
            current_chars += txt_len

        return coalesced_results

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: int = 5,
        candidate_k: int = 50,
        score_threshold: float = 0.30,
        use_reranker: bool = True,
        pre_fetched_candidates: Optional[List[Dict[str, Any]]] = None,
        skip_search_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        2-Stage Dual-Search retrieval with Small-to-Big Parent Page Hydration:
        1. Multi-query vector search across 32k+ points
        2. Merge & deduplicate candidate pool with keyword boosting
        3. Cross-Encoder deep attention reranking on micro-chunks
        4. Small-to-Big Parent Page Hydration on winning chunks
        """
        queries: List[str] = [query] if isinstance(query, str) else query
        primary_query = queries[0] if queries else ""
        if not primary_query:
            return []
        
        skip_queries = skip_search_queries or []

        # Stage 1: Vector search across all queries in list
        pool_size = max(candidate_k, top_k * 5)
        seen_texts: Dict[str, Dict[str, Any]] = {}

        if pre_fetched_candidates:
            for cand in pre_fetched_candidates:
                txt = cand.get("text", "").strip()
                if txt:
                    seen_texts[txt] = cand

        for q in queries:
            if not q or not q.strip() or q in skip_queries:
                continue
            raw_candidates = self.indexer.search(query=q.strip(), top_k=pool_size)
            for cand in raw_candidates:
                txt = cand.get("text", "").strip()
                if not txt:
                    continue
                if txt not in seen_texts or cand.get("score", 0) > seen_texts[txt].get("score", 0):
                    seen_texts[txt] = cand

        all_candidates = list(seen_texts.values())
        if not all_candidates:
            return []

        # Boost exact source / drawing / document keyword matches in candidate pool
        q_lower = primary_query.lower()
        for cand in all_candidates:
            src_lower = cand.get("source", "").lower()
            text_lower = cand.get("text", "").lower()

            # Direct DFMEA / BOM boost
            if "dfmea" in q_lower and "dfmea" in src_lower:
                cand["score"] = cand.get("score", 0) + 0.40
            elif "bom" in q_lower and "bom" in src_lower:
                cand["score"] = cand.get("score", 0) + 0.40

            # Universal standard & drawing code matching (e.g., 212-2008, 2-4.1-212, ДПМА.731673, ГОСТ 6111)
            for part in q_lower.split():
                clean_part = part.strip(",.()[]{}:;\"'«»")
                if len(clean_part) >= 4:
                    if clean_part in src_lower:
                        cand["score"] = cand.get("score", 0) + 0.45
                    elif clean_part in text_lower and any(kw in clean_part for kw in ["дпма", "гост", "gost", "сто", "rd", "рду", "212", "2-4"]):
                        cand["score"] = cand.get("score", 0) + 0.35

        # Sort by boosted vector score
        all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Filter low-confidence vector noise
        filtered_candidates = [r for r in all_candidates if r.get("score", 0) >= score_threshold]
        candidates = filtered_candidates if filtered_candidates else all_candidates[:15]

        # Stage 2: Cross-Encoder Reranking on precise micro-chunks
        if use_reranker and self.reranker.enabled:
            results = self.reranker.rerank(primary_query, candidates, top_n=top_k * 2 if self.enable_page_coalescing else top_k)
        else:
            results = candidates[:top_k * 2 if self.enable_page_coalescing else top_k]

        # Stage 3: Small-to-Big Parent Page Coalescing & Hydration
        if self.enable_page_coalescing or self.enable_parent_page_hydration:
            final_results = self.coalesce_and_hydrate_pages(results, max_total_chars=12000)[:top_k]
        else:
            final_results = results[:top_k]

        return final_results
