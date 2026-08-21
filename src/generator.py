"""
LLM Generator & Prompt Assembly Module.
Streams tokens via Ollama and formats structured engineering citations.
"""
import os
import json
import requests
from typing import List, Dict, Any, Generator, Optional

SYSTEM_PROMPT_TEMPLATE = """You are an expert Chief Engineer and Normative Technical Consultant in gas pipelines and valve equipment.
Your task is to provide an accurate, factually grounded engineering answer strictly based on the provided technical standards and specifications.

CRITICAL RULES:
1. Grounding: Answer ONLY based on the facts present in the [CONTEXT CHUNKS]. Do not invent values or specifications.
2. Citation Format: Whenever citing a rule, standard, or value, explicitly cite the source using: [SOURCE #X: Document_Name, Page Y].
3. Missing Information: If the provided context does not contain the answer, explicitly state: "The provided knowledge base documents do not contain information to answer this question."
4. Structure: Structure your response with markdown headings, tables for technical data, and bullet points.

[CONTEXT CHUNKS]:
{context}
"""

class OllamaGenerator:
    """Generates streaming technical answers from context using local LLMs."""
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.base_url: str = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name: str = model_name or os.getenv("LLM_MODEL", "qwen3.6:35b")
        self.temperature: float = temperature

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Assembles prompt with numbered context chunks."""
        context_parts = []
        for i, c in enumerate(context_chunks, 1):
            src = c.get("source", "Standard")
            page = c.get("page", "1")
            text = c.get("text", "").strip()
            context_parts.append(f"--- [CHUNK #{i}] (Source: {src}, Page: {page}) ---\n{text}")

        context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
        return f"{system_content}\n\nEngineer Question: {query}\nExpert Answer:"

    def generate_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams tokens from Ollama formatted as Server-Sent Events (SSE)."""
        prompt = self.build_prompt(query, context_chunks)
        target_model = model_override or self.model_name

        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9
            }
        }

        # Yield source metadata
        sources_meta = [
            {
                "index": i,
                "source": c.get("source"),
                "page": c.get("page"),
                "score": c.get("final_score", c.get("score")),
                "verification_link": f"/api/documents/preview/{c.get('source')}#page={c.get('page', 1)}"
            }
            for i, c in enumerate(context_chunks, 1)
        ]
        yield f"event: sources\ndata: {json.dumps(sources_meta, ensure_ascii=False)}\n\n"

        url = f"{self.base_url}/api/generate"
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                if data.get("done", False):
                    break
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
