"""
Adaptive PDF Parser with Vision-Language Model OCR fallback.
"""
import os
import io
import base64
import subprocess
import requests
from typing import List
from pypdf import PdfReader
from src.chunker import TextChunk
from src.parsers.base import BaseDocumentParser

class PdfParser(BaseDocumentParser):
    """Parses digital and scanned PDF standards."""
    def __init__(self, ollama_url: str = None, vision_model: str = "minicpm-v:latest"):
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.vision_model = vision_model

    def _ocr_page_via_vlm(self, pdf_path: str, page_num: int) -> str:
        """Executes Vision-Language OCR using local MiniCPM-V."""
        try:
            cmd = ["pdftoppm", "-png", "-r", "150", "-f", str(page_num), "-l", str(page_num), pdf_path]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            img_bytes = proc.stdout
            if not img_bytes:
                return ""
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            payload = {
                "model": self.vision_model,
                "prompt": "Transcribe this technical standard page into clean markdown preserving tables and section titles verbatim.",
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1}
            }
            resp = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        return ""

    def parse(self, file_path: str) -> List[TextChunk]:
        reader = PdfReader(file_path)
        chunks: List[TextChunk] = []
        filename = os.path.basename(file_path)

        for page_idx, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if len(text) < 30:
                ocr_text = self._ocr_page_via_vlm(file_path, page_idx)
                if ocr_text:
                    text = ocr_text

            if text:
                meta = {
                    "source": filename,
                    "page": page_idx,
                    "format": "pdf",
                    "chunk_id": f"{filename}_p{page_idx}"
                }
                chunks.append(TextChunk(text=text, metadata=meta))

        return chunks
