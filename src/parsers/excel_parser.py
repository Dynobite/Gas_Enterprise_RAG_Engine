"""
Hierarchical Excel DFMEA and BOM Spreadsheet Parser.
"""
import os
import openpyxl
from typing import List
from src.chunker import TextChunk
from src.parsers.base import BaseDocumentParser

class ExcelParser(BaseDocumentParser):
    """Parses Excel spreadsheets with cell unmerging and header propagation."""
    def parse(self, file_path: str) -> List[TextChunk]:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        chunks: List[TextChunk] = []
        filename = os.path.basename(file_path)

        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue

            headers = [str(c or "").strip() for c in rows[0]]
            for row_idx, row in enumerate(rows[1:], 2):
                row_items = []
                for h, val in zip(headers, row):
                    if val is not None and str(val).strip():
                        row_items.append(f"{h}: {str(val).strip()}")
                
                if row_items:
                    row_text = f"Worksheet: {sheet.title} | Row {row_idx}: " + "; ".join(row_items)
                    meta = {
                        "source": filename,
                        "sheet": sheet.title,
                        "row": row_idx,
                        "format": "xlsx",
                        "chunk_id": f"{filename}_{sheet.title}_r{row_idx}"
                    }
                    chunks.append(TextChunk(text=row_text, metadata=meta))

        return chunks
