"""
Embedding Client module interfacing with Ollama API for BGE-M3 and Nomic embeddings.
"""

import os
import json
from typing import List

try:
    import requests as _requests

    def _http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

except ImportError:
    import urllib.request as _urllib_request

    def _http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:  # type: ignore[misc]
        data = json.dumps(payload).encode("utf-8")
        req = _urllib_request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with _urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class OllamaEmbeddingClient:
    """HTTP client to obtain dense vector embeddings from an Ollama-hosted model."""

    # Known embedding dimensions per model prefix
    _DIMENSION_MAP: dict = {
        "bge-m3": 1024,
        "nomic-embed": 768,
    }

    def __init__(
        self,
        base_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model: str = os.getenv("EMBEDDING_MODEL", "bge-m3"),
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.dimension: int = self._resolve_dimension(model)

    @classmethod
    def _resolve_dimension(cls, model: str) -> int:
        """Return embedding dimension based on model name prefix."""
        model_lower = model.lower()
        for prefix, dim in cls._DIMENSION_MAP.items():
            if prefix in model_lower:
                return dim
        return 768  # safe default for unknown models

    def get_embedding(self, text: str) -> List[float]:
        """Fetch dense vector embedding for a single text string."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            data = _http_post_json(url, payload, timeout=60)
            embedding: List[float] = data.get("embedding", [])
            if len(embedding) != self.dimension:
                print(
                    f"[WARN] Unexpected embedding dimension: got {len(embedding)}, "
                    f"expected {self.dimension} for model '{self.model}'."
                )
            return embedding if embedding else [0.0] * self.dimension
        except Exception as exc:
            print(
                f"[ERROR] Embedding generation failed for model '{self.model}' "
                f"at {url}: {exc}"
            )
            return [0.0] * self.dimension

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed a list of texts sequentially.

        Args:
            texts: List of raw text strings to embed.
            batch_size: Reserved for future parallel implementation (unused).

        Returns:
            List of embedding vectors in the same order as ``texts``.
        """
        embeddings: List[List[float]] = []
        total = len(texts)
        for idx, text in enumerate(texts, start=1):
            embeddings.append(self.get_embedding(text))
            if idx % 10 == 0 or idx == total:
                print(f"  [EMBED] Processed {idx}/{total} chunks...")
        return embeddings
