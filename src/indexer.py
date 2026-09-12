"""
Qdrant Vector Indexer Module.
Initializes embedded Qdrant vector database, generates BGE-M3 embeddings,
and stores document chunks with rich metadata payloads.
"""

import os
import sys
import json
import uuid
from typing import List, Dict, Any, Optional

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_UPSERT_BATCH_SIZE = 100  # Max points per Qdrant upsert call to avoid memory spikes

from src.embeddings import OllamaEmbeddingClient

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

class QdrantVectorIndexer:
    def __init__(
        self,
        db_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        collection_name: str = "gas_rag_standards",
        embedding_model: str = "bge-m3",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.db_path = db_path or os.getenv("VECTOR_DB_DIR", os.path.join(project_root, "vector_db"))
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name
        self.embed_client = OllamaEmbeddingClient(base_url=ollama_url, model=embedding_model)
        self.vector_dim = self.embed_client.dimension
        
        os.makedirs(self.db_path, exist_ok=True)
        
        if HAS_QDRANT:
            # 1. Attempt connection to standalone Qdrant Daemon (High Performance, No File Locks)
            connected = False
            if self.qdrant_url:
                try:
                    self.client = QdrantClient(url=self.qdrant_url, check_compatibility=False, timeout=5)
                    self.client.get_collections()
                    connected = True
                    print(f"[QDRANT] Connected to standalone Qdrant Server Daemon at {self.qdrant_url}")
                except Exception as exc:
                    print(f"[WARN] Could not connect to Qdrant Server at {self.qdrant_url} ({exc}). Falling back to local mode.")
            
            # 2. Fallback to local embedded mode if server unreachable
            if not connected:
                self.client = QdrantClient(path=self.db_path)
                print(f"[QDRANT] Running in local embedded mode (storage: {self.db_path})")

            self._ensure_collection()
        else:
            self.client = None
            print("[WARN] qdrant-client not installed. Please install via `pip install qdrant-client`.")

    def _ensure_collection(self) -> None:
        """Create or verify the Qdrant collection with cosine similarity."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                print(f"[QDRANT] Creating collection '{self.collection_name}' (dim={self.vector_dim}, metric=COSINE)...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )
            else:
                print(f"[QDRANT] Collection '{self.collection_name}' verified online.")
        except Exception as e:
            print(f"[WARN] Could not verify collection '{self.collection_name}': {e}")


    def index_chunks(self, chunks: List[Any], recreate_collection: bool = False) -> int:
        """Directly compute BGE-M3 embeddings for chunk objects and upsert into Qdrant."""
        if not HAS_QDRANT or not self.client:
            print("[ERROR] Cannot index: qdrant-client is not available.")
            return 0

        total_chunks = len(chunks)
        print(f"[QDRANT] Indexing {total_chunks} chunk objects into '{self.collection_name}'...")

        points = []
        for idx, item in enumerate(chunks, start=1):
            if hasattr(item, "text"):
                text = item.text
                metadata = getattr(item, "metadata", {})
            else:
                text = item.get("text", "")
                metadata = item.get("metadata", {})

            embedding = self.embed_client.get_embedding(text)
            chunk_unique_name = f"{metadata.get('source', '')}_{metadata.get('sheet', '')}_{metadata.get('page', '')}_{idx}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_name))

            payload = {
                "text": text,
                **metadata
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        total_upserted = 0
        for batch_start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[batch_start: batch_start + _UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total_upserted += len(batch)

        print(f"[SUCCESS] Successfully indexed {total_upserted} chunks into Qdrant collection '{self.collection_name}'!")
        return total_upserted

    def index_chunks_file(self, chunks_json_path: str = None) -> int:
        """Read chunks from chunks.json, compute BGE-M3 embeddings, and upsert into Qdrant."""
        if not HAS_QDRANT or not self.client:
            print("[ERROR] Cannot index: qdrant-client is not available.")
            return 0

        path = chunks_json_path or os.path.join(project_root, "data", "processed", "chunks.json")
        if not os.path.exists(path):
            print(f"[ERROR] Chunks file not found: {path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        total_chunks = len(chunks_data)
        print(f"[QDRANT] Indexing {total_chunks} chunks from '{path}' into '{self.collection_name}'...")

        points = []
        for idx, item in enumerate(chunks_data, start=1):
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            
            # Compute embedding
            embedding = self.embed_client.get_embedding(text)
            
            # Create unique point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, metadata.get("chunk_id", f"chunk_{idx}")))
            
            payload = {
                "text": text,
                **metadata
            }
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

            if idx % 10 == 0 or idx == total_chunks:
                print(f"  [INDEXING] Processed {idx}/{total_chunks} embeddings...")

        # Batch upsert to avoid memory spikes with large datasets
        total_upserted = 0
        for batch_start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[batch_start: batch_start + _UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total_upserted += len(batch)
            print(f"  [UPSERT] Committed {total_upserted}/{len(points)} points...")

        print(f"[SUCCESS] Successfully upserted {total_upserted} points into Qdrant collection '{self.collection_name}'!")
        return total_upserted

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on Qdrant collection."""
        if not HAS_QDRANT or not self.client:
            return []

        query_vector = self.embed_client.get_embedding(query)
        
        # Qdrant client query points
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points

        results = []
        for scored_point in search_result:
            meta = {k: v for k, v in scored_point.payload.items() if k != "text"}
            results.append({
                "score": round(float(scored_point.score), 4),
                "text": scored_point.payload.get("text", ""),
                "metadata": meta,
                **meta
            })

        return results

if __name__ == "__main__":
    indexer = QdrantVectorIndexer()
    indexer.index_chunks_file()
