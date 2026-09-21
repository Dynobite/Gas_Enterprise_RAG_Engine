"""
Pre-Retrieval Engineering Co-Pilot & Dynamic Slot Clarifier (Plan B).
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional

class QueryClarifier:
    """Detects missing engineering parameters and suggests interactive clarification chips."""
    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen3.6:35b")

    def clarify(self, draft_query: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Returns deficiency analysis and 1-click grounded query alternatives."""
        if len(draft_query.split()) > 6 or re.search(r'\d+[\.\-]\d+', draft_query):
            return {
                "is_specific_enough": True,
                "clarification_question": None,
                "options": []
            }

        return {
            "is_specific_enough": False,
            "clarification_question": "Please specify the component, standard, or operating condition:",
            "options": [
                {
                    "label": "Pipeline Valve Testing (GOST 33257)",
                    "full_query": f"{draft_query} according to GOST 33257 pipeline valve testing"
                },
                {
                    "label": "Material Specifications (Steel 09G2S)",
                    "full_query": f"{draft_query} for Steel 09G2S low temperature service"
                },
                {
                    "label": "DFMEA Risk Analysis (Ball Valves)",
                    "full_query": f"{draft_query} in ball valve DFMEA risk matrix"
                }
            ]
        }
