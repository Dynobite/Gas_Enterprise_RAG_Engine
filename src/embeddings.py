"""
Dense Embeddings Client Module.
Computes high-dimensional vector embeddings via Ollama (BGE-M3).
"""
import os
import requests
from typing import List, Optional

class OllamaEmbeddingClient:
    """Client for generating dense vector embeddings via local Ollama instance."""
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url: str = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name: str = model_name or os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.timeout: int = timeout

    def get_embedding(self, text: str) -> List[float]:
        """Generates a dense embedding vector for the provided text."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError("Ollama response missing 'embedding' field")
            return embedding
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama embedding service: {e}")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text strings."""
        return [self.get_embedding(t) for t in texts]
