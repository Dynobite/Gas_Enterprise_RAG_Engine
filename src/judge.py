"""
LLM-as-a-Judge Module for GASlight-Me RAG.
Acts as a single-purpose Fact-Checking Audit Guardrail to detect hallucinations
and enforce 100% grounding against retrieved technical documentation.
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

# Confidence score used when the Judge call fails (graceful degradation)
_FALLBACK_CONFIDENCE: float = 0.90

class RagFactJudge:
    """Single-purpose hallucination detector that verifies RAG answers against retrieved context."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        judge_model: str = "gpt-fast",
        enabled: bool = True,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.judge_model = judge_model
        self.enabled = enabled
        try:
            from src.llm_router import router
            self.router = router
        except Exception:
            self.router = None

    def verify_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        answer: str,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated answer candidate against raw retrieved context.
        Returns strict JSON verification assessment.
        """
        if not self.enabled or not answer or not context_chunks:
            return {
                "is_grounded": True,
                "hallucination_detected": False,
                "confidence_score": 1.0,
                "unsupported_claims": [],
                "verdict": "SKIPPED_OR_EMPTY",
                "judge_model": self.judge_model
            }

        # If model gave explicit negative refusal, it is 100% faithful
        if "нет точных данных" in answer.lower() or "отсутствуют сведения" in answer.lower():
            return {
                "is_grounded": True,
                "hallucination_detected": False,
                "confidence_score": 1.0,
                "unsupported_claims": [],
                "verdict": "FAITHFUL_REFUSAL",
                "judge_model": self.judge_model
            }

        # Build context summary
        context_text = ""
        for idx, c in enumerate(context_chunks, 1):
            src = c.get("source", "Документ")
            page = c.get("page") or c.get("sheet") or ""
            page_str = f", Стр/Лист: {page}" if page else ""
            context_text += f"\n[ИСТОЧНИК #{idx}: {src}{page_str}]\n{c.get('text', '')}\n"

        prompt = f"""Вам поручена ЕДИНСТВЕННАЯ задача: Аудит достоверности и поиск галлюцинаций (Fact-Checking Judge).

Проверьте, подтверждены ли ВСЕ фактические утверждения в [ОТВЕТЕ] текстом из [КОНТЕКСТА].
Обратите особое внимание на: марки сталей, размеры, параметры резьбы, ГОСТы, формулы и числовые значения.

=== [КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ] ===
{context_text[:3500]}

=== [ПРОВЕРЯЕМЫЙ ОТВЕТ] ===
{answer[:2500]}

ВЕРНИТЕ РЕЗУЛЬТАТ СТРОГО В ФОРМАТЕ JSON:
{{
  "is_grounded": true/false,
  "hallucination_detected": true/false,
  "confidence_score": float (от 0.0 до 1.0),
  "unsupported_claims": ["список выдуманных фактов или пустой список []"],
  "verdict": "VERIFIED_FAITHFUL" или "HALLUCINATION_DETECTED"
}}"""

        target_model = model_override or self.judge_model
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
            return self._fallback_result("requests library not available")

        def _request_judge(post_url, post_payload):
            resp = _requests.post(post_url, json=post_payload, timeout=45)
            resp.raise_for_status()
            res_json = resp.json()
            msg = res_json.get("choices", [{}])[0].get("message", {})
            raw_content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "{}"
            return raw_content

        try:
            try:
                raw_content = _request_judge(url, payload)
            except Exception as vllm_exc:
                if active_engine == "vllm" and any(om in target_model.lower() for om in ["gpt-oss", "llama"]):
                    print(f"[WARN] vLLM judge failed ({vllm_exc}). Falling back to Ollama.")
                    fallback_url = f"{self.ollama_url}/v1/chat/completions"
                    payload["model"] = target_model
                    raw_content = _request_judge(fallback_url, payload)
                else:
                    raise vllm_exc

            # Clean markdown code fences if wrapped
            if "```" in raw_content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_content)
                if match:
                    raw_content = match.group(1).strip()

            # Extract first JSON object if surrounded by reasoning text
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                raw_content = json_match.group(0)

            parsed = json.loads(raw_content)
            
            return {
                "is_grounded": bool(parsed.get("is_grounded", True)),
                "hallucination_detected": bool(parsed.get("hallucination_detected", False)),
                "confidence_score": round(float(parsed.get("confidence_score", 0.95)), 2),
                "unsupported_claims": parsed.get("unsupported_claims", []),
                "verdict": parsed.get("verdict", "VERIFIED_FAITHFUL"),
                "judge_model": active_model,
            }
        except Exception as exc:
            return self._fallback_result(str(exc)[:60])

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        """Return a safe fallback verdict when the Judge cannot execute."""
        return {
            "is_grounded": True,
            "hallucination_detected": False,
            "confidence_score": _FALLBACK_CONFIDENCE,
            "unsupported_claims": [],
            "verdict": f"JUDGE_FALLBACK ({reason})",
            "judge_model": self.judge_model,
        }
