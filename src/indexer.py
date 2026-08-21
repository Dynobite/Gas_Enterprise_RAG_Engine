"""
Qdrant Indexer Module.
Creates and manages HNSW vector collections in pure-Rust Qdrant daemon.
"""
import os
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.embeddings import OllamaEmbeddingClient
from src.chunker import TextChunk

_UPSERT_BATCH_SIZE = 100

class QdrantIndexer:
    """Manages document vector indexing in Qdrant Server."""
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_dim: int = 1024,
        embed_client: Optional[OllamaEmbeddingClient] = None
    ):
        self.host: str = host or os.getenv("QDRANT_HOST", "localhost")
        self.port: int = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name: str = collection_name or os.getenv("COLLECTION_NAME", "gas_rag_standards")
        self.vector_dim: int = vector_dim
        self.embed_client: OllamaEmbeddingClient = embed_client or OllamaEmbeddingClient()
        self.client: QdrantClient = QdrantClient(host=self.host, port=self.port, timeout=60.0)

    def init_collection(self, recreate: bool = False):
        """Initializes or resets the target Qdrant collection with HNSW index."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
        )

    def index_chunks(self, chunks: List[TextChunk], batch_size: int = _UPSERT_BATCH_SIZE) -> int:
        """Embeds and uploads text chunks to Qdrant in batches."""
        if not chunks:
            return 0

        self.init_collection(recreate=False)
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            points: List[PointStruct] = []
            for idx, c in enumerate(batch):
                emb = self.embed_client.get_embedding(c.text)
                chunk_id = c.metadata.get("chunk_id", str(uuid.uuid4()))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))
                payload = {"text": c.text, **c.metadata}
                points.append(PointStruct(id=point_id, vector=emb, payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)
            total_indexed += len(points)

        return total_indexed
