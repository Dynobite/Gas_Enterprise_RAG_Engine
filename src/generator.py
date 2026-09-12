"""
Generator Module for GAS-RAG.
Formats retrieved context into engineering prompts and calls Qwen 3.6 35B / DeepSeek R1 on Ollama.
"""

import os
import json
from typing import List, Dict, Any, Optional, Generator

try:
    import requests as _requests
    _HAS_REQUESTS = True

    def _http_post_json(url: str, payload: dict, timeout: int = 120) -> dict:
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

except ImportError:
    _HAS_REQUESTS = False
    _requests = None  # type: ignore[assignment]
    import urllib.request as _urllib_request

    def _http_post_json(url: str, payload: dict, timeout: int = 120) -> dict:  # type: ignore[misc]
        data = json.dumps(payload).encode("utf-8")
        req = _urllib_request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with _urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

class GasRagGenerator:
    def __init__(
        self,
        ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        default_model: str = "qwen3.6:35b",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = default_model
        try:
            from src.llm_router import router
            self.router = router
        except Exception:
            self.router = None

    @staticmethod
    def _build_sources(context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source metadata list from retrieved context chunks."""
        from config import resolve_sto_part_and_page
        sources: List[Dict[str, Any]] = []
        for c in context_chunks:
            meta = c.get("metadata", {})
            raw_source = meta.get("source", "Документ")
            raw_page = meta.get("page", 1)
            resolved_source, resolved_page = resolve_sto_part_and_page(raw_source, raw_page)
            vlink = f"/api/documents/preview/{resolved_source}?page={resolved_page}#page={resolved_page}"

            sources.append({
                "source": resolved_source,
                "page": resolved_page,
                "original_source": raw_source,
                "original_page": raw_page,
                "sheet": meta.get("sheet"),
                "doc_type": meta.get("doc_type"),
                "score": c.get("score"),
                "verification_link": vlink,
            })
        return sources

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], deep_reasoning: bool = False) -> str:
        """Format context chunks with source headers for citation tracking."""
        from config import resolve_sto_part_and_page
        context_blocks = []
        for idx, item in enumerate(context_chunks, start=1):
            meta = item.get("metadata", {})
            source = meta.get("source", "Документ")
            page = meta.get("page", 1)
            resolved_source, resolved_page = resolve_sto_part_and_page(source, page)
            sheet = meta.get("sheet", "")
            doc_type = meta.get("doc_type", "Стандарт")
            location = f"Стр. {resolved_page}" if page else f"Лист: {sheet}"
            text = item.get("text", "").strip()

            header = f"[ИСТОЧНИК #{idx}: {doc_type} | Файл: {resolved_source} | {location}]"
            context_blocks.append(f"{header}\n{text}")

        full_context = "\n\n".join(context_blocks)
        # Cap full context at 4200 chars (~2200-2500 tokens) to guarantee ample room (4000+ tokens) for generation within 8192 limit
        if len(full_context) > 4200:
            full_context = full_context[:4200] + "\n...[контекст оптимизирован по размеру]..."

        rule_5 = (
            "5. Предварительный инженерный анализ нормативных требований проводите в блоке рассуждений, после чего сформируйте подробный итоговый технический ответ со ссылками."
            if deep_reasoning
            else "5. Сразу же выводите готовый технический ответ на русском языке без предварительных рассуждений."
        )

        system_prompt = (
            "Вы — ведущий инженерный ИИ-ассистент департамента R&D (GAS-RAG).\n"
            "Ваша задача — предоставлять точные, строго аргументированные технические ответы на основе нормативных стандартов, "
            "спецификаций BOM, протоколов испытаний и анализов DFMEA.\n\n"
            "ПРАВИЛА ОТВЕТА:\n"
            "1. Полагайтесь ТОЛЬКО на предоставленный контекст ниже. Не придумывайте факты.\n"
            "2. В ответе ОБЯЗАТЕЛЬНО делайте ссылки на источники в формате: [ИСТОЧНИК #X: Название, Стр. Y].\n"
            "3. Если точных данных нет в контексте, прямо напишите: 'В предоставленных документах базы знаний нет точных данных по данному вопросу.'\n"
            "4. Форматируйте итоговый ответ структурированно, с маркерами или таблицами, на грамотном русском техническом языке.\n"
            f"{rule_5}"
        )

        return f"{system_prompt}\n\n=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===\n{full_context}\n\n=== ВОПРОС ИНЖЕНЕРА ===\n{query}\n\n=== ОТВЕТ ИНЖЕНЕРА-АССИСТЕНТА ==="

    def _compute_safe_max_tokens(self, prompt: str, target_max: int = 4096) -> int:
        """
        Compute guaranteed safe max_tokens for vLLM / Ollama.
        Queries /tokenize on vLLM if available for exact token count,
        or uses ultra-conservative ratio (1.6 chars/token) as fallback.
        Guarantees prompt_tokens + max_tokens <= model_limit - 64.
        """
        model_limit = 8192
        exact_tokens = None

        if self.router and self.router.is_vllm_online() and _HAS_REQUESTS:
            try:
                tokenize_url = f"{self.router.vllm_url}/tokenize"
                resp = _requests.post(tokenize_url, json={"model": "qwen3.6:35b", "prompt": prompt}, timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json()
                    exact_tokens = data.get("count")
                    model_limit = data.get("max_model_len", 8192)
            except Exception:
                pass

        if exact_tokens is None:
            # Conservative fallback: 1.6 chars/token for dense technical Cyrillic text
            exact_tokens = int(len(prompt) / 1.6) + 128

        safe_budget = model_limit - exact_tokens - 64
        return max(512, min(target_max, safe_budget))

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None,
        deep_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Send chat request to Ollama and format response with verification links."""
        target_model = model_override or self.default_model
        prompt = self.build_prompt(query, context_chunks, deep_reasoning=deep_reasoning)
        safe_max_tokens = self._compute_safe_max_tokens(prompt, target_max=4096)
        
        if self.router:
            target = self.router.resolve_target(target_model)
            url = f"{target['base_url']}/chat/completions"
            active_model = target["model"]
            active_engine = target["engine"]
        else:
            url = f"{self.ollama_url}/v1/chat/completions"
            active_model = target_model
            active_engine = "ollama"

        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": safe_max_tokens,
            "chat_template_kwargs": {"enable_thinking": deep_reasoning}
        }

        try:
            data = _http_post_json(url, payload, timeout=120)
            answer_text: str = data.get("choices", [{}])[0].get("message", {}).get("content", "Не удалось сформировать ответ.")
            suggestions = self.generate_suggestions(query=query, answer=answer_text, context_chunks=context_chunks, model_override=target_model)
            return {
                "query": query,
                "answer": answer_text,
                "model_used": target_model,
                "sources": self._build_sources(context_chunks),
                "suggestions": suggestions,
            }
        except Exception as exc:
            return {
                "query": query,
                "answer": f"[ОШИБКА] Не удалось выполнить генерацию через Ollama ({self.ollama_url}): {exc}",
                "model_used": target_model,
                "sources": [],
                "suggestions": [],
            }

    def generate_suggestions(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None,
    ) -> List[str]:
        """Generate 3 concise, highly relevant engineering follow-up questions based on the answer and context (Plan C)."""
        if not answer or "не удалось" in answer.lower():
            return []

        target_model = model_override or self.default_model
        prompt = f"""Вам поручена роль: Инженерный консультант R&D.
На основе исходного вопроса инженера и сформированного ответа, сформулируйте РОВНО 3 кратких, полезных и логичных наводящих вопроса (Follow-up / Guiding Questions) для углубления инженерного анализа (по деталям, смежным ГОСТам, испытаниям или анализу рисков DFMEA).

Вопрос инженера: {query}
Ответ системы: {answer[:600]}

ВЕРНИТЕ РЕЗУЛЬТАТ СТРОГО В ФОРМАТЕ JSON:
{{
  "suggestions": [
    "Краткий вопрос 1",
    "Краткий вопрос 2",
    "Краткий вопрос 3"
  ]
}}"""

        if self.router:
            target = self.router.resolve_target(target_model)
            url = f"{target['base_url']}/chat/completions"
            active_model = target["model"]
        else:
            url = f"{self.ollama_url}/v1/chat/completions"
            active_model = target_model

        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 512,
            "response_format": {"type": "json_object"}
        }
        try:
            data = _http_post_json(url, payload, timeout=60)
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}").strip()
            
            # Clean markdown code fences if wrapped
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            parsed = json.loads(raw_content)
            items = parsed.get("suggestions", [])
            if isinstance(items, list) and items:
                res = [str(s).strip() for s in items[:3] if str(s).strip()]
                if len(res) >= 2:
                    return res
        except Exception as exc:
            print(f"[WARN] Failed to parse follow-up suggestions ({exc}). Using context-aware fallbacks.")

        # Fallback intelligent suggestions based on engineering context
        return [
            f"Какие требования к испытаниям на герметичность по этой позиции?",
            f"Показать сопряженные детали и материалы по спецификации BOM",
            f"Какие риски и виды отказов зафиксированы в анализе DFMEA?"
        ]


    def generate_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        model_override: Optional[str] = None,
        deep_reasoning: bool = False
    ) -> Generator[str, None, None]:
        """Yield Server-Sent Events (SSE) streaming tokens using Ollama API."""
        target_model = model_override or self.default_model
        prompt = self.build_prompt(query, context_chunks, deep_reasoning=deep_reasoning)
        safe_max_tokens = self._compute_safe_max_tokens(prompt, target_max=4096)
        
        # Dual-engine routing: primary vLLM, fallback Ollama
        if self.router:
            target = self.router.resolve_target(target_model)
            url = f"{target['base_url']}/chat/completions"
            active_model = target["model"]
            active_engine = target["engine"]
        else:
            url = f"{self.ollama_url}/v1/chat/completions"
            active_model = target_model
            active_engine = "ollama"

        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": safe_max_tokens,
            "chat_template_kwargs": {"enable_thinking": deep_reasoning}
        }

        # 1. Send sources metadata first
        yield f"event: sources\ndata: {json.dumps(self._build_sources(context_chunks), ensure_ascii=False)}\n\n"

        # 2. Stream tokens in real time
        def _stream_from(stream_url, stream_payload):
            with _requests.post(stream_url, json=stream_payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            chunk_json = json.loads(line_str[6:])
                            choices = chunk_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                                if token:
                                    yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        try:
            yield from _stream_from(url, payload)
        except Exception as exc:
            if active_engine == "vllm" and any(om in target_model.lower() for om in ["gpt-oss", "llama"]):
                print(f"[WARN] vLLM stream failed ({exc}). Falling back to Ollama.")
                fallback_url = f"{self.ollama_url}/v1/chat/completions"
                payload["model"] = target_model
                try:
                    yield from _stream_from(fallback_url, payload)
                except Exception as fallback_exc:
                    yield f"event: error\ndata: {json.dumps({'error': str(fallback_exc)}, ensure_ascii=False)}\n\n"
            else:
                err_detail = getattr(exc, 'response', None)
                err_body = err_detail.text if err_detail is not None else ""
                print(f"[ERROR] Stream generation failed: {exc} | detail: {err_body}")
                yield f"event: error\ndata: {json.dumps({'error': f'Ошибка генерации: {exc} | {err_body}'}, ensure_ascii=False)}\n\n"


