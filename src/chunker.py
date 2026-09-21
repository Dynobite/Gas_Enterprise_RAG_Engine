"""
Text Chunker Module.
Splits text into chunks by word count with configurable overlap.
"""
from typing import List, Dict, Any, Optional

class TextChunk:
    """Represents a discrete chunk of text extracted from a document."""
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text: str = text
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        page = self.metadata.get("page", "?")
        return f"<TextChunk len={len(self.text)} source={source} page={page}>"


class SlidingWindowChunker:
    """Splits plain text into overlapping word windows."""
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

    def chunk(self, text: str, base_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Splits text into sliding window chunks."""
        words = text.split()
        if not words:
            return []

        meta = base_metadata.copy() if base_metadata else {}
        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_meta = meta.copy()
            chunk_meta["chunk_index"] = len(chunks) + 1
            chunks.append(TextChunk(text=chunk_text, metadata=chunk_meta))

        return chunks
