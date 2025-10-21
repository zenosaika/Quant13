from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from src.config import load_config


class LLMClient:
    def __init__(self) -> None:
        config = load_config()
        self.base_url = config["openrouter"].get("base_url", "https://openrouter.ai/api/v1")
        self.model = config["openrouter"].get("model")
        api_key_env = config["openrouter"].get("api_key_env", "OPENROUTER_API_KEY")
        self.api_key = os.getenv(api_key_env)

    def chat(self, messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.7) -> str:
        if not self.api_key:
            return self._fallback_response(messages)
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic heuristic used when no API key is available."""
        last_message = messages[-1]["content"] if messages else ""
        if "winning_argument" in last_message or "Trade Thesis" in last_message:
            return json.dumps({
                "winning_argument": "Bullish",
                "conviction_level": "Medium",
                "summary": "Fallback thesis: bullish argument favored due to stronger quantitative support.",
                "key_evidence": [
                    "Volatility conditions supportive.",
                    "Technical momentum remains constructive.",
                ],
            })
        if "trade" in last_message.lower():
            return json.dumps({
                "strategy": "Call Debit Spread",
                "direction": "Bullish",
                "expiration": "45D",
                "strikes": [100.0, 110.0],
                "notes": "Fallback trade suggestion in absence of model access.",
            })
        return "Model unavailable; defaulting to safe response."


def get_llm_client() -> LLMClient:
    return LLMClient()
