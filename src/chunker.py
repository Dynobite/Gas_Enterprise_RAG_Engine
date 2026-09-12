"""
Chunker module applying sliding window with overlap to document texts.
"""

from typing import List, Dict, Any
from src.parsers.base import DocumentChunk

class SlidingWindowChunker:
    def __init__(self, chunk_size_words: int = 400, chunk_overlap_words: int = 80):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def chunk_documents(self, doc_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Takes raw page/section chunks and splits large ones into overlapping windows."""
        final_chunks = []
        
        for doc in doc_chunks:
            if not doc.text or not doc.text.strip():
                continue  # Skip empty/whitespace-only document sections

            words = doc.text.split()
            
            # If the chunk is already small enough (e.g. single page or small table), keep as is
            if len(words) <= self.chunk_size_words:
                chunk_id = f"{doc.metadata.get('source', 'doc')}_p{doc.metadata.get('page', 1)}_c0"
                meta = dict(doc.metadata)
                meta["chunk_id"] = chunk_id
                meta["word_count"] = len(words)
                final_chunks.append(DocumentChunk(text=doc.text, metadata=meta))
                continue

            # Apply sliding window
            step = max(1, self.chunk_size_words - self.chunk_overlap_words)
            chunk_sub_idx = 0
            
            for i in range(0, len(words), step):
                window_words = words[i:i + self.chunk_size_words]
                chunk_text = " ".join(window_words)
                
                chunk_id = f"{doc.metadata.get('source', 'doc')}_p{doc.metadata.get('page', 1)}_c{chunk_sub_idx}"
                meta = dict(doc.metadata)
                meta["chunk_id"] = chunk_id
                meta["chunk_index"] = chunk_sub_idx
                meta["word_count"] = len(window_words)
                
                final_chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
                chunk_sub_idx += 1
                
                if i + self.chunk_size_words >= len(words):
                    break

        return final_chunks
