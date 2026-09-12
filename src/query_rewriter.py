"""
Query Rewriter & Prompt Engineer Module for GASlight-Me RAG.
Normalizes conversational, slang, or noisy user queries into structured technical queries
while strictly preserving exact ГОСТ numbers, drawing designations, and part numbers.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    _requests = None  # type: ignore[assignment]

# Ensure project root in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class GasRagQueryRewriter:
    """Intelligent query optimizer that translates informal engineering queries into formal technical search prompts."""

    def __init__(
        self,
        ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model: str = "gpt-fast",
        enabled: bool = True,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        try:
            from src.llm_router import router
            self.router = router
        except Exception:
            self.router = None

    def rewrite_query(self, user_query: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze user query, strip conversational noise, normalize terminology,
        and generate an optimized technical search query for Qdrant/BGE-M3.

        Returns:
            Dict with:
                - original_query: str
                - optimized_query: str
                - search_queries: List[str] (both queries for dual-search)
                - technical_entities: List[str]
                - is_rewritten: bool
        """
        clean_input = (user_query or "").strip()
        if not self.enabled or not clean_input:
            return self._fallback_result(clean_input)

        # Skip rewriting for very short exact alphanumeric lookups (e.g. "ГОСТ 6111-52", "ДПМА.731673.011")
        tokens = clean_input.split()
        if len(tokens) <= 3 and any(char.isdigit() for char in clean_input):
            return {
                "original_query": clean_input,
                "optimized_query": clean_input,
                "search_queries": [clean_input],
                "technical_entities": [clean_input],
                "is_rewritten": False,
            }

        prompt = f"""Вам поручена роль: Инженерный оптимизатор поисковых запросов для RAG-базы стандартов (ГОСТ, ISO, BOM, DFMEA).

ЗАДАЧА:
Преобразуйте пользовательский запрос в четкий, очищенный от словесного шума технический поисковый запрос.

ПРАВИЛА:
1. Удалите слова-паразиты и разговорные фразы («подскажи», «слушай», «чо там по», «как насчет», «срочно надо»).
2. Замените сленг на ГОСТ/инженерные термины («внешка» -> «корпус», «80-ка» -> «DN80 / Ду80», «резинки» -> «уплотнения»).
3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО изменять, искажать или выдумывать буквенно-цифровые шифры, номера чертежей, ГОСТы и артикулы (например, «ДПМА.731673.011», «ГОСТ 6111-52», «09Г2С»). Переносите их БЕЗ ИЗМЕНЕНИЙ!
4. Выделите ключевые технические сущности (марки сталей, типы арматуры, ГОСТы, узлы).

Пользовательский запрос:
"{clean_input}"

ВЕРНИТЕ РЕЗУЛЬТАТ СТРОГО В ФОРМАТЕ JSON:
{{
  "optimized_query": "краткий технический запрос для векторного поиска",
  "technical_entities": ["сущность1", "сущность2"],
  "is_rewritten": true
}}"""

        target_model = model_override or self.model
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
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"}
        }

        if not _HAS_REQUESTS or _requests is None:
            return self._fallback_result(clean_input)

        def _request_rewrite(post_url, post_payload):
            resp = _requests.post(post_url, json=post_payload, timeout=60)
            resp.raise_for_status()
            res_json = resp.json()
            msg = res_json.get("choices", [{}])[0].get("message", {})
            return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "{}"

        try:
            try:
                raw_content = _request_rewrite(url, payload)
            except Exception as vllm_exc:
                if active_engine == "vllm" and any(om in target_model.lower() for om in ["gpt-oss", "llama"]):
                    print(f"[WARN] vLLM query_rewriter failed ({vllm_exc}). Falling back to Ollama.")
                    fallback_url = f"{self.ollama_url}/v1/chat/completions"
                    payload["model"] = target_model
                    raw_content = _request_rewrite(fallback_url, payload)
                else:
                    raise vllm_exc

            # Strip <think>...</think> tags if present
            raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

            # Clean markdown code fences if wrapped
            if "```" in raw_content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_content)
                if match:
                    raw_content = match.group(1).strip()

            # Extract first JSON object if surrounded by extra text
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                raw_content = json_match.group(0)

            parsed = json.loads(raw_content)


            opt_query = parsed.get("optimized_query", "").strip()
            if not opt_query:
                opt_query = clean_input

            entities = parsed.get("technical_entities", [])
            if not isinstance(entities, list):
                entities = []

            # Dual-search list: original query + optimized query (deduplicated)
            search_queries = [clean_input]
            if opt_query.lower() != clean_input.lower():
                search_queries.append(opt_query)

            return {
                "original_query": clean_input,
                "optimized_query": opt_query,
                "search_queries": search_queries,
                "technical_entities": entities,
                "is_rewritten": opt_query.lower() != clean_input.lower(),
            }

        except Exception as exc:
            print(f"[WARN] Query rewriting failed ({exc}). Using original query.")
            return self._fallback_result(clean_input)

    def _fallback_result(self, raw_query: str) -> Dict[str, Any]:
        """Safe fallback when rewriter is disabled or offline."""
        return {
            "original_query": raw_query,
            "optimized_query": raw_query,
            "search_queries": [raw_query],
            "technical_entities": [],
            "is_rewritten": False,
        }
