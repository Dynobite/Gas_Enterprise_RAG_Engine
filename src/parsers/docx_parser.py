"""
DOCX Technical Report Parser.
"""
import os
import mammoth
from typing import List
from src.chunker import TextChunk
from src.parsers.base import BaseDocumentParser

class DocxParser(BaseDocumentParser):
    """Parses DOCX documents into clean markdown text."""
    def parse(self, file_path: str) -> List[TextChunk]:
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text = result.value.strip()

        chunks: List[TextChunk] = []
        if text:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs, 1):
                meta = {
                    "source": filename,
                    "section": idx,
                    "format": "docx",
                    "chunk_id": f"{filename}_sec{idx}"
                }
                chunks.append(TextChunk(text=p, metadata=meta))

        return chunks
