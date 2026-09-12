import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Generator

logger = logging.getLogger(__name__)

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error


class LLMEngineRouter:
    """
    Intelligent dual-engine LLM router for Gas_RAG.
    Directs inference to high-performance vLLM (port 8001) when online,
    and transparently falls back to Ollama (port 11434) if vLLM is offline or erroring.
    """

    def __init__(
        self,
        vllm_url: Optional[str] = None,
        ollama_url: Optional[str] = None,
        health_check_ttl: float = 10.0,
    ) -> None:
        self.vllm_url = (vllm_url or os.getenv("VLLM_HOST", "http://localhost:8001")).rstrip("/")
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.health_check_ttl = health_check_ttl

        self._vllm_online: bool = False
        self._last_health_check: float = 0.0
        self._vllm_models: List[str] = []

    def is_vllm_online(self, force_check: bool = False) -> bool:
        """Check if vLLM is available, cached with TTL."""
        now = time.time()
        if not force_check and (now - self._last_health_check) < self.health_check_ttl:
            return self._vllm_online

        self._last_health_check = now
        try:
            url = f"{self.vllm_url}/v1/models"
            if _HAS_REQUESTS:
                resp = requests.get(url, timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json()
                    self._vllm_models = [m.get("id") for m in data.get("data", [])]
                    self._vllm_online = True
                    return True
            else:
                req = _urllib_request.Request(url, headers={"User-Agent": "GasRAG-Router"})
                with _urllib_request.urlopen(req, timeout=1.5) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        self._vllm_models = [m.get("id") for m in data.get("data", [])]
                        self._vllm_online = True
                        return True
        except Exception:
            pass

        self._vllm_online = False
        self._vllm_models = []
        return False

    def resolve_target(self, requested_model: str) -> Dict[str, Any]:
        """
        Determine target endpoint and model name.
        If vLLM is online and has a corresponding model, routes to vLLM.
        Otherwise routes to Ollama.
        """
        if self.is_vllm_online():
            # Match model if loaded in vLLM
            vllm_model = None
            for m in self._vllm_models:
                if requested_model in m or m in requested_model:
                    vllm_model = m
                    break
            
            # If vLLM has only one model loaded, default to it for general/internal tasks
            # but allow explicit alternate models (like gpt-oss, llama-vision) to reach Ollama
            if not vllm_model and len(self._vllm_models) == 1:
                if not any(om in requested_model.lower() for om in ["vision", "llama", "gpt-oss"]):
                    vllm_model = self._vllm_models[0]

            if vllm_model:
                return {
                    "engine": "vllm",
                    "base_url": f"{self.vllm_url}/v1",
                    "model": vllm_model,
                }

        # Fallback to Ollama
        return {
            "engine": "ollama",
            "base_url": f"{self.ollama_url}/v1",
            "model": requested_model,
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        response_format: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Execute non-streaming chat completion with automatic fallback from vLLM to Ollama.
        """
        target = self.resolve_target(model)
        url = f"{target['base_url']}/chat/completions"
        payload = {
            "model": target["model"],
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if response_format:
            payload["response_format"] = response_format

        # Attempt primary engine
        try:
            if _HAS_REQUESTS:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                res = resp.json()
                res["_engine"] = target["engine"]
                return res
            else:
                body = json.dumps(payload).encode("utf-8")
                req = _urllib_request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json", "User-Agent": "GasRAG-Router"},
                )
                with _urllib_request.urlopen(req, timeout=timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    res["_engine"] = target["engine"]
                    return res
        except Exception as exc:
            if target["engine"] == "vllm":
                logger.warning(f"[ROUTER] vLLM request failed ({exc}). Falling back to Ollama.")
                # Mark vLLM as offline and fallback
                self._vllm_online = False
                fallback_url = f"{self.ollama_url}/v1/chat/completions"
                payload["model"] = model
                if _HAS_REQUESTS:
                    resp = requests.post(fallback_url, json=payload, timeout=timeout)
                    resp.raise_for_status()
                    res = resp.json()
                    res["_engine"] = "ollama"
                    return res
                else:
                    body = json.dumps(payload).encode("utf-8")
                    req = _urllib_request.Request(
                        fallback_url,
                        data=body,
                        headers={"Content-Type": "application/json", "User-Agent": "GasRAG-Router"},
                    )
                    with _urllib_request.urlopen(req, timeout=timeout) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        res["_engine"] = "ollama"
                        return res
            raise exc

    def get_status(self) -> Dict[str, Any]:
        """Return engine availability and telemetry."""
        is_online = self.is_vllm_online()
        return {
            "vllm_online": is_online,
            "vllm_url": self.vllm_url,
            "vllm_models": self._vllm_models,
            "ollama_url": self.ollama_url,
            "active_engine": "vllm" if is_online else "ollama",
        }


# Singleton instance
router = LLMEngineRouter()
