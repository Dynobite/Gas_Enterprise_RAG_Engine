"""
Text and Markdown Parser for plain text standards, specifications, and notes.
"""

import os
from typing import List, Dict, Any
from src.parsers.base import BaseParser, DocumentChunk

class TextParser(BaseParser):
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith((".txt", ".md", ".csv", ".json", ".log"))

    def parse(self, file_path: str) -> List[DocumentChunk]:
        chunks = []
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            doc_type = "GOST Standard" if "ГОСТ" in filename.upper() else "Text Standard / Note"
            
            metadata = {
                "source": filename,
                "file_path": os.path.abspath(file_path),
                "page": 1,
                "total_pages": 1,
                "doc_type": doc_type,
                "format": "txt",
                "verification_link": f"file://{os.path.abspath(file_path).replace(os.sep, '/')}"
            }
            chunks.append(DocumentChunk(text=content, metadata=metadata))

        except Exception as e:
            print(f"Error parsing Text file {file_path}: {e}")

        return chunks
