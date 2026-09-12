"""
Two-Stage Pre-Search & Dynamic Slot-Filling Query Clarifier with LRU Warm Cache.
Performs 15ms dense Qdrant pre-retrieval to extract factual document anchors,
then uses a fast LLM (gpt-oss:20b) to generate a natural clarifying question with 4 grounded options in ~1.5s.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

class GasRagQueryClarifier:
    def __init__(
        self,
        ollama_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model: str = "qwen3.6:35b",
        retriever: Optional[Any] = None,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.retriever = retriever
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max = 500
        try:
            from src.llm_router import router
            self.router = router
        except Exception:
            self.router = None

    def clarify(self, draft_query: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform Two-Stage Grounded Slot-Filling analysis on draft query.
        1. Check in-memory LRU Warm Cache (<1ms)
        2. Perform 15ms Dense Pre-Search in Qdrant DB
        3. Fast LLM Slot-Filling Analysis (~1.5s)
        """
        clean_input = draft_query.strip()
        if not clean_input:
            return self._empty_result()

        target_model = model_override or self.model
        cache_key = f"{clean_input.lower()}::{target_model}"

        # 1. Warm In-Memory Cache (< 1 ms response)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check if already highly specific (contains exact part codes, BOM/DFMEA identifiers or standard numbers)
        has_drawing_code = bool(re.search(r'[А-ЯA-Z]{3,4}\.[\dA-ZА-Я]{4,8}\.[\dA-ZА-Я]{2,6}', clean_input, re.IGNORECASE))
        has_doc_keyword = bool(re.search(r'(DFMEA|BOM|спецификаци|паспорт|руководств|чертеж|таблиц)', clean_input, re.IGNORECASE))
        has_gost_code = bool(re.search(r'(ГОСТ|СТО|РД|ТУ|ISO|DIN)\s*[\d\-.]+', clean_input, re.IGNORECASE))
        has_dn_pn = bool(re.search(r'(DN|ДУ|PN|РУ)\s*\d+', clean_input, re.IGNORECASE))
        is_long_query = len(clean_input.split()) >= 6

        is_already_specific = has_drawing_code or has_doc_keyword or (has_gost_code and has_dn_pn) or (is_long_query and (has_gost_code or has_drawing_code))

        if is_already_specific:
            res = {
                "is_specific_enough": True,
                "clarifying_question": "Запрос содержит точные спецификации и готов к прямому поиску по базе знаний.",
                "options": [],
                "enhanced_example": clean_input,
                "original_query": clean_input,
            }
            self._set_cache(cache_key, res)
            return res

        # 2. Stage 1: Fast 15ms Dense Qdrant Pre-Search for factual grounding
        grounded_facts = ""
        if self.retriever is not None:
            try:
                raw_matches = self.retriever.retrieve(clean_input, top_k=3)
                if raw_matches:
                    fact_lines = []
                    for doc in raw_matches[:3]:
                        src = doc.get("source", "Документ")
                        text_snip = doc.get("text", "")[:140].replace("\n", " ").strip()
                        fact_lines.append(f"• [{src}]: {text_snip}...")
                    if fact_lines:
                        grounded_facts = "\nФактические документы, найденные в базе Qdrant:\n" + "\n".join(fact_lines) + "\n"
            except Exception:
                pass

        # 3. Stage 2: Fast LLM In-Context Slot-Filling
        prompt = f"""Вам поручена роль: Инженер-консультант R&D по трубопроводной арматуре и стандартам Газпрома.
Пользователь ввел черновик запроса к базе знаний.
{grounded_facts}
Черновик запроса: "{clean_input}"

ВАША ЗАДАЧА:
1. Выявите недостающий инженерный слот (Узел/Деталь? Марка стали? Климатика? Вид испытаний? Стандарт?).
2. Сформулируйте РОВНО ОДИН естественный, вежливый уточняющий вопрос (например: "Температура эксплуатации какого узла или материала вас интересует?").
3. Предложите РОВНО 4 конкретных, заземленных на базу знаний варианта (label + full_query).

ВЕРНИТЕ РЕЗУЛЬТАТ СТРОГО В ФОРМАТЕ JSON:
{{
  "is_specific_enough": false,
  "clarifying_question": "Уточняющий вопрос инженеру?",
  "options": [
    {{
      "label": "Краткий вариант 1",
      "full_query": "Развернутый инженерный запрос 1"
    }},
    {{
      "label": "Краткий вариант 2",
      "full_query": "Развернутый инженерный запрос 2"
    }},
    {{
      "label": "Краткий вариант 3",
      "full_query": "Развернутый инженерный запрос 3"
    }},
    {{
      "label": "Краткий вариант 4",
      "full_query": "Развернутый инженерный запрос 4"
    }}
  ]
}}"""

        if self.router:
            target = self.router.resolve_target(target_model)
            url = f"{target['base_url']}/chat/completions"
            payload = {
                "model": target["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }
        else:
            url = f"{self.ollama_url}/v1/chat/completions"
            payload = {
                "model": target_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }

        if not _HAS_REQUESTS or _requests is None:
            fallback = self._fallback_result(clean_input)
            self._set_cache(cache_key, fallback)
            return fallback

        try:
            resp = _requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            res_json = resp.json()
            if "choices" in res_json and res_json["choices"]:
                raw_content = res_json["choices"][0].get("message", {}).get("content", "{}").strip()
            else:
                raw_content = res_json.get("message", {}).get("content", "{}").strip()

            # Strip <think>...</think> tags if present
            raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

            # Clean markdown code fences if wrapped
            if "```" in raw_content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_content)
                if match:
                    raw_content = match.group(1).strip()

            # Extract first JSON object
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                raw_content = json_match.group(0)

            parsed = json.loads(raw_content)
            question = parsed.get("clarifying_question", "").strip()
            options = parsed.get("options", [])

            if question and isinstance(options, list) and len(options) >= 2:
                clean_opts = []
                for opt in options[:4]:
                    if isinstance(opt, dict) and "label" in opt:
                        lbl = str(opt.get("label", "")).strip()
                        fq = str(opt.get("full_query", lbl)).strip()
                        if lbl:
                            clean_opts.append({"label": lbl, "full_query": fq})
                    elif isinstance(opt, str) and opt.strip():
                        clean_opts.append({"label": opt.strip(), "full_query": f"{clean_input} {opt.strip()}"})

                if clean_opts:
                    final_res = {
                        "is_specific_enough": False,
                        "clarifying_question": question,
                        "options": clean_opts,
                        "original_query": clean_input,
                    }
                    self._set_cache(cache_key, final_res)
                    return final_res
        except Exception as exc:
            print(f"[WARN] Fast Clarifier generation failed ({exc}). Using context-adaptive fallback.")

        fallback = self._fallback_result(clean_input)
        self._set_cache(cache_key, fallback)
        return fallback

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        if len(self._cache) >= self._cache_max:
            # Remove oldest key
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "is_specific_enough": False,
            "clarifying_question": "Введите черновик вопроса для анализа (например: 'температура эксплуатации' или 'испытания затвора').",
            "options": [],
            "original_query": "",
        }

    def _fallback_result(self, clean_input: str) -> Dict[str, Any]:
        """Context-adaptive heuristic fallback."""
        q_lower = clean_input.lower()

        if "температур" in q_lower:
            return {
                "is_specific_enough": False,
                "clarifying_question": "Температура эксплуатации какого именно элемента или исполнения вас интересует?",
                "options": [
                    {
                        "label": "Корпус в исполнении ХЛ1 / УХЛ1 (до -60°C)",
                        "full_query": "Какова минимальная и максимальная допустимая температура эксплуатации корпуса арматуры в исполнении ХЛ1 по СТО Газпром?",
                    },
                    {
                        "label": "Уплотнения штока и затвора (PTFE / FKM)",
                        "full_query": "Какой температурный диапазон эксплуатации уплотнительных материалов из PTFE и эластомеров?",
                    },
                    {
                        "label": "Корпусная сталь 09Г2С по ГОСТ 19281",
                        "full_query": "Каковы предельные температуры эксплуатации деталей из стали 09Г2С по ГОСТ 19281?",
                    },
                    {
                        "label": "Запорная арматура по СТО Газпром 2-4.1",
                        "full_query": "Какие требования к температуре эксплуатации запорной арматуры установлены в СТО Газпром 2-4.1?",
                    },
                ],
                "original_query": clean_input,
            }
        elif "резьб" in q_lower or "зазор" in q_lower:
            return {
                "is_specific_enough": False,
                "clarifying_question": "О каком типе резьбового соединения или стандарте идет речь?",
                "options": [
                    {
                        "label": "Коническая резьба по ГОСТ 6111-52",
                        "full_query": "Каковы требования к размерам, допускам и зазорам конической дюймовой резьбы по ГОСТ 6111-52?",
                    },
                    {
                        "label": "Метрическая резьба шпилек фланцев ГОСТ 33259",
                        "full_query": "Каковы требования к резьбе крепежа фланцевых соединений по ГОСТ 33259-2015?",
                    },
                    {
                        "label": "Ходовая резьба шпинделя затвора",
                        "full_query": "Каковы требования к ходовой трапецеидальной резьбе шпинделя по спецификации BOM?",
                    },
                    {
                        "label": "Резьбовые пробки и штуцеры",
                        "full_query": "Каковы требования к резьбовым пробкам и дренажным штуцерам по СТО Газпром?",
                    },
                ],
                "original_query": clean_input,
            }
        elif "испытан" in q_lower or "давлен" in q_lower:
            return {
                "is_specific_enough": False,
                "clarifying_question": "Какой именно вид испытаний на давление вас интересует?",
                "options": [
                    {
                        "label": "Гидравлические испытания корпуса на прочность (1.5 PN)",
                        "full_query": "Каковы параметры и выдержка при гидравлическом испытании корпуса на прочность 1.5 PN по ГОСТ 33257?",
                    },
                    {
                        "label": "Испытания затвора на герметичность (1.1 PN, Класс А)",
                        "full_query": "Каковы нормы утечек и требования к испытаниям на герметичность затвора Класс А по ГОСТ 33257-2015?",
                    },
                    {
                        "label": "Испытания на огнестойкость (Fire-Safe)",
                        "full_query": "Какие требования предъявляются к испытаниям запорной арматуры на огнестойкость по СТО Газпром?",
                    },
                    {
                        "label": "Пневматические испытания воздухом",
                        "full_query": "В каких случаях и по каким нормам проводятся пневматические испытания затвора воздухом?",
                    },
                ],
                "original_query": clean_input,
            }
        else:
            return {
                "is_specific_enough": False,
                "clarifying_question": f"Какой именно узел, стандарт или исполнение требуется уточнить для '{clean_input}'?",
                "options": [
                    {
                        "label": "Затвор дисковый DN80 PN160 (BOM ДПМА)",
                        "full_query": f"{clean_input} для затвора дискового DN80 PN160 по спецификации BOM ДПМА",
                    },
                    {
                        "label": "Требования по СТО Газпром 2-4.1",
                        "full_query": f"{clean_input} по требованиям СТО Газпром 2-4.1-212",
                    },
                    {
                        "label": "Нормы по ГОСТ 33257-2015",
                        "full_query": f"{clean_input} по правилам испытаний ГОСТ 33257-2015",
                    },
                    {
                        "label": "Материалы корпусных деталей (09Г2С)",
                        "full_query": f"{clean_input} для деталей из стали 09Г2С по ГОСТ 19281",
                    },
                ],
                "original_query": clean_input,
            }
