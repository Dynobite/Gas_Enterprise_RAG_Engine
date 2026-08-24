"""
PDF Parser with Adaptive OllamaOCR for digital standards, scanned documents, and technical manuals.
Extracts native text when available, and automatically routes scanned/image-only pages
to Ollama Vision Model (minicpm-v) for high-fidelity Markdown transcription.
"""

import os
import re
import base64
import subprocess
import tempfile
import urllib.request
import json
from typing import List, Dict, Any, Optional
from src.parsers.base import BaseParser, DocumentChunk

class PDFParser(BaseParser):
    def __init__(
        self,
        min_char_threshold: int = 30,
        ollama_url: str = "http://localhost:11434",
        vision_model: str = "llama3.2-vision:11b",
        enable_ocr: bool = True
    ):
        self.min_char_threshold = min_char_threshold
        self.ollama_url = ollama_url.rstrip("/")
        self.vision_model = vision_model
        self.enable_ocr = enable_ocr

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def detect_standard_type(self, filename: str, first_page_text: str) -> str:
        text = (filename + " " + first_page_text[:1000]).upper()
        if "ГОСТ" in text or "GOST" in text:
            return "GOST Standard"
        elif "ISO" in text:
            return "ISO Standard"
        elif "DIN" in text:
            return "DIN Standard"
        elif "СТО" in text or "ГАЗПРОМ" in text:
            return "Gazprom Standard (СТО)"
        elif "РД" in text or "РУКОВОДЯЩИЙ ДОКУМЕНТ" in text:
            return "Guiding Document (РД)"
        elif "ИССЛЕДОВАНИЙ" in text or "ОТЧЕТ" in text or "REPORT" in text:
            return "Research Report"
        elif "ПРОТОКОЛ" in text or "АКТ" in text:
            return "Protocol / Act"
        return "Technical PDF"

    def clean_text(self, text: str) -> str:
        """Clean line breaks, fix hyphens, and normalize multiple spaces."""
        if not text:
            return ""
        # Strip any accidental Asian/CJK character hallucinations (e.g. from Chinese-biased models)
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+', ' ', text)
        # Fix soft line hyphenation (e.g. стан-\nдарт -> стандарт)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Normalize excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize multiple spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def ocr_page_via_vision(self, pdf_path: str, page_num: int) -> str:
        """
        Renders a PDF page to PNG at 200 DPI and transcribes it using an Ollama Vision Model.
        Uses pure greedy decoding (temperature: 0.0) and strict Cyrillic constraints.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = os.path.join(tmp_dir, f"page_{page_num}")
            cmd = ["pdftoppm", "-png", "-r", "200", "-f", str(page_num), "-l", str(page_num), pdf_path, prefix]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"[OCR ERROR] pdftoppm failed on page {page_num}: {e}")
                return ""

            png_files = [f for f in os.listdir(tmp_dir) if f.endswith(".png")]
            if not png_files:
                return ""
            
            png_path = os.path.join(tmp_dir, png_files[0])
            with open(png_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode("utf-8")

            prompt = (
                "Ты — высокоточный OCR-транскрибатор технических документов и стандартов. "
                "Преобразуй этот скан страницы в структурированный Markdown на русском языке. "
                "Точно передай весь русский текст, номера разделов, таблицы, параметры и формулы. "
                "СТРОГОЕ ПРАВИЛО: Используй только русский язык и стандартные латинские обозначения. "
                "Любые китайские или иероглифические символы СТРОГО ЗАПРЕЩЕНЫ. "
                "Если на странице нет текста, верни пустую строку."
            )

            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [b64_img],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1500
                }
            }

            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    extracted = res.get("response", "").strip()
                    if "отсутствует содержимое" in extracted.lower() or "белый фон" in extracted.lower():
                        return ""
                    # Strip CJK unicode ranges
                    cleaned_extracted = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f]+', ' ', extracted).strip()
                    return cleaned_extracted
            except Exception as exc:
                print(f"[OCR WARN] Vision model '{self.vision_model}' failed on page {page_num}: {exc}")
                return ""

    def parse(self, file_path: str) -> List[DocumentChunk]:
        chunks = []
        filename = os.path.basename(file_path)

        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            total_pages = len(reader.pages)

            # Extract first page to detect document category
            first_page_text = reader.pages[0].extract_text() or "" if total_pages > 0 else ""
            doc_type = self.detect_standard_type(filename, first_page_text)

            print(f"[PDF PARSER] Processing '{filename}' ({total_pages} pages)...")

            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = self.clean_text(raw_text)

                # Check if page is scanned / image-only and needs OllamaOCR
                if len(cleaned) < self.min_char_threshold and self.enable_ocr:
                    print(f"  [OllamaOCR] Page {page_idx}/{total_pages} has no text layer. Transcribing with {self.vision_model}...")
                    ocr_text = self.ocr_page_via_vision(file_path, page_idx)
                    cleaned = self.clean_text(ocr_text)

                if not cleaned:
                    continue

                metadata = {
                    "source": filename,
                    "file_path": os.path.abspath(file_path),
                    "page": page_idx,
                    "total_pages": total_pages,
                    "doc_type": doc_type,
                    "format": "pdf",
                    "verification_link": f"http://localhost:8000/api/documents/{filename}#page={page_idx}"
                }

                chunks.append(DocumentChunk(text=cleaned, metadata=metadata))

        except Exception as e:
            print(f"[ERROR] Error parsing PDF {file_path}: {e}")

        return chunks
