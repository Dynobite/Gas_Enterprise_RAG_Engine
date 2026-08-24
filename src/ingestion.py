"""
Unified Document Ingestion Pipeline.
Loads raw documents from data/raw, applies format-specific parsers,
extracts structured Markdown (Stage 1), chunks text, and saves artifacts to data/processed/.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.parsers.base import DocumentChunk
from src.parsers.pdf_parser import PDFParser
from src.parsers.docx_parser import DocxParser
from src.parsers.excel_parser import ExcelParser
from src.parsers.text_parser import TextParser
from src.chunker import SlidingWindowChunker

class DocumentIngestionPipeline:
    def __init__(self, raw_dir: str = None, processed_dir: str = None):
        self.raw_dir = raw_dir or os.path.join(project_root, "data", "raw")
        self.processed_dir = processed_dir or os.path.join(project_root, "data", "processed")
        self.markdown_dir = os.path.join(self.processed_dir, "markdown")
        os.makedirs(self.markdown_dir, exist_ok=True)
        
        self.parsers = [
            PDFParser(),
            DocxParser(),
            ExcelParser(),
            TextParser()
        ]
        self.chunker = SlidingWindowChunker(chunk_size_words=400, chunk_overlap_words=80)

    def extract_raw_chunks(self, file_path: str) -> List[DocumentChunk]:
        """Extract unchunked document sections/pages from matching parser."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser.parse(file_path)
        print(f"[WARN] No parser found for {file_path}")
        return []

    def build_structured_markdown(self, filename: str, raw_chunks: List[DocumentChunk]) -> str:
        """Convert extracted chunks into a unified, high-fidelity Markdown document."""
        if not raw_chunks:
            return f"# {filename}\n\n*(Документ пуст или не содержит извлекаемого текста)*\n"

        lines = [f"# 📄 {filename}\n\n"]
        lines.append(f"> **Источник**: `{filename}` | **Всего фрагментов/страниц**: `{len(raw_chunks)}`\n\n---\n\n")

        # Group by page or section
        current_page = None
        current_sheet = None

        for idx, chunk in enumerate(raw_chunks):
            meta = chunk.metadata or {}
            page = meta.get("page")
            sheet = meta.get("sheet_name")

            if sheet and sheet != current_sheet:
                current_sheet = sheet
                lines.append(f"\n## 📊 Лист Excel: {sheet}\n\n")
            elif page and page != current_page:
                current_page = page
                lines.append(f"\n## 📄 Страница {page}\n\n")

            lines.append(f"{chunk.text.strip()}\n\n")

        return "".join(lines)

    def save_markdown_artifact(self, filename: str, markdown_content: str, chunks: List[DocumentChunk]) -> str:
        """Save structured markdown and metadata cache to data/processed/markdown/."""
        os.makedirs(self.markdown_dir, exist_ok=True)
        base_name = os.path.basename(filename)
        md_file = os.path.join(self.markdown_dir, f"{base_name}.md")
        meta_file = os.path.join(self.markdown_dir, f"{base_name}.meta.json")

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Count tables and characters
        table_count = len(re.findall(r'\|.+\|', markdown_content))
        meta_data = {
            "filename": base_name,
            "char_count": len(markdown_content),
            "word_count": len(markdown_content.split()),
            "chunks_count": len(chunks),
            "table_lines": table_count,
            "has_tables": table_count > 0,
            "created_at": str(os.path.getmtime(md_file) if os.path.exists(md_file) else "")
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        return md_file

    def get_document_markdown(self, filename: str) -> Dict[str, Any]:
        """
        Get or dynamically generate the Stage 1 parsed Markdown for a document.
        Returns dict with raw_markdown, metadata, and chunk stats.
        """
        base_name = os.path.basename(filename)
        md_file = os.path.join(self.markdown_dir, f"{base_name}.md")
        meta_file = os.path.join(self.markdown_dir, f"{base_name}.meta.json")
        raw_file = os.path.join(self.raw_dir, base_name)

        if os.path.exists(md_file) and os.path.exists(meta_file):
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "status": "cached",
                "filename": base_name,
                "markdown": md_content,
                "metadata": meta
            }

        # If not exact match, resolve filename against raw directory
        if not os.path.exists(raw_file) and os.path.exists(self.raw_dir):
            all_raw = os.listdir(self.raw_dir)
            matched = None
            for rf in all_raw:
                if rf.lower() == base_name.lower():
                    matched = rf
                    break
            if not matched:
                # Suffix / stem match
                stem = os.path.splitext(base_name)[0].lower()
                ext = os.path.splitext(base_name)[1].lower()
                for rf in all_raw:
                    rf_stem = os.path.splitext(rf)[0].lower()
                    rf_ext = os.path.splitext(rf)[1].lower()
                    if ext == rf_ext and (stem in rf_stem or rf_stem in stem):
                        matched = rf
                        break
            if matched:
                raw_file = os.path.join(self.raw_dir, matched)

        if not os.path.exists(raw_file):
            return {
                "status": "error",
                "filename": base_name,
                "markdown": f"# Ошибка: Файл `{base_name}` не найден в `data/raw/`",
                "metadata": {}
            }

        raw_chunks = self.extract_raw_chunks(raw_file)
        md_content = self.build_structured_markdown(base_name, raw_chunks)
        self.save_markdown_artifact(base_name, md_content, raw_chunks)

        return {
            "status": "generated",
            "filename": base_name,
            "markdown": md_content,
            "metadata": {
                "filename": base_name,
                "char_count": len(md_content),
                "word_count": len(md_content.split()),
                "chunks_count": len(raw_chunks),
            }
        }

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """Route file to matching parser, generate Markdown artifact, and chunk for indexing."""
        raw_chunks = self.extract_raw_chunks(file_path)
        if not raw_chunks:
            return []

        filename = os.path.basename(file_path)
        md_content = self.build_structured_markdown(filename, raw_chunks)
        self.save_markdown_artifact(filename, md_content, raw_chunks)

        # Excel is already chunked by row/component
        if file_path.lower().endswith((".xlsx", ".xls")):
            return raw_chunks
        return self.chunker.chunk_documents(raw_chunks)

    def run(self) -> List[DocumentChunk]:
        """Ingest all documents in raw_dir, generate Markdown artifacts, and save chunks.json."""
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.markdown_dir, exist_ok=True)
        all_chunks: List[DocumentChunk] = []

        if not os.path.exists(self.raw_dir):
            print(f"Raw directory '{self.raw_dir}' does not exist.")
            return []

        files = sorted(os.listdir(self.raw_dir))
        print(f"[INFO] Starting Stage 1 Ingestion & Markdown Generation for {len(files)} files...")

        stats = {}
        for f in files:
            file_path = os.path.join(self.raw_dir, f)
            if not os.path.isfile(file_path):
                continue

            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
            stats[f] = len(chunks)
            print(f"  [OK] {f} -> {len(chunks)} chunks (Markdown generated)")

        # Save to data/processed/chunks.json
        output_file = os.path.join(self.processed_dir, "chunks.json")
        data_to_save = [c.to_dict() for c in all_chunks]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"\n[DONE] Ingestion Complete!")
        print(f"[INFO] Total Chunks Extracted: {len(all_chunks)}")
        print(f"[INFO] Saved to: {os.path.abspath(output_file)}")
        print(f"[INFO] Markdown Directory: {os.path.abspath(self.markdown_dir)}")
        
        return all_chunks

if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.run()
