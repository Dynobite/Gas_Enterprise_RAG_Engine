"""
Query Rewriter Module (Plan A).
Expands technical acronyms and standards codes before vector retrieval.
"""
import os
import re
from typing import List, Dict, Any, Optional
from src.embeddings import OllamaEmbeddingClient

class QueryRewriter:
    """Expands engineering queries with standard codes and technical terminology."""
    def __init__(self, embed_client: Optional[OllamaEmbeddingClient] = None):
        self.embed_client: OllamaEmbeddingClient = embed_client or OllamaEmbeddingClient()

    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """Expands query and extracts target standards."""
        cleaned = query.strip()
        search_queries = [cleaned]

        # Extract standard numbers like 2-4.1-212
        std_match = re.search(r'(\d+[\.\-]\d+[\.\-]\d+[\.\-]?\d*)', cleaned)
        if std_match:
            std_code = std_match.group(1)
            search_queries.append(f"Standard {std_code} requirements specification")

        return {
            "original_query": cleaned,
            "search_queries": search_queries,
            "is_expanded": len(search_queries) > 1
        }
