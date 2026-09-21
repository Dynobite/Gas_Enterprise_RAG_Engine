"""
Document Ingestion Pipeline Module.
Dispatches raw documents to format-specific parsers.
"""
import os
from typing import List
from src.chunker import TextChunk
from src.parsers.pdf_parser import PdfParser
from src.parsers.excel_parser import ExcelParser
from src.parsers.docx_parser import DocxParser

class IngestionPipeline:
    """Dispatches document files to dedicated parsers based on extension."""
    def __init__(self):
        self.pdf_parser = PdfParser()
        self.excel_parser = ExcelParser()
        self.docx_parser = DocxParser()

    def process_file(self, file_path: str) -> List[TextChunk]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.pdf_parser.parse(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self.excel_parser.parse(file_path)
        elif ext in [".docx", ".doc"]:
            return self.docx_parser.parse(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            filename = os.path.basename(file_path)
            return [TextChunk(text=text, metadata={"source": filename, "format": "txt"})]
        return []
