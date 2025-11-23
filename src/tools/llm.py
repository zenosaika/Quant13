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
            "HTTP-Referer": "https://github.com/quant13/trading-system",
            "X-Title": "Quant13 Options Trading System",
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            return self._fallback_response(messages)

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic heuristic used when no API key is available."""
        last_message = messages[-1]["content"] if messages else ""
        if "MD&A section" in last_message:
            return json.dumps({
                "tone": "neutral",
                "performance_drivers": ["Revenue growth stabilized", "Cost controls improving margins"],
                "forward_looking": ["Management guides mid-single digit growth", "Focus on operating leverage"],
            })
        if "Risk Factors" in last_message:
            return json.dumps([
                {"risk": "Macroeconomic slowdown", "category": "Market", "rationale": "Demand softness could weigh on revenue."},
                {"risk": "Regulatory scrutiny", "category": "Regulatory", "rationale": "Ongoing investigations may lead to fines."},
            ])
        if "\"transcript\"" in last_message:
            return json.dumps({
                "winning_argument": "Bullish",
                "conviction_level": "Medium",
                "summary": "Fallback thesis: bullish argument favored due to stronger quantitative support.",
                "key_evidence": [
                    "Volatility conditions supportive.",
                    "Technical momentum remains constructive.",
                ],
            })
        if "\"articles\"" in last_message:
            return json.dumps({
                "articles": [
                    {
                        "title": "Company announces product update",
                        "publisher": "Newswire",
                        "published_at": None,
                        "sentiment_score": 0.1,
                        "rationale": "Incrementally positive but limited detail.",
                    }
                ],
                "overall_sentiment_score": 0.1,
                "overall_summary": "Headlines skew slightly positive with limited impact.",
            })
        if "\"indicators\"" in last_message:
            return json.dumps({
                "technical_bias": "neutral",
                "primary_trend": "Price consolidating around key moving averages.",
                "momentum": "RSI and MACD indicate balanced momentum.",
                "volatility_levels": "Bollinger Bands show moderate compression near median.",
                "key_levels": {"support": "50-day SMA", "resistance": "Recent swing high"},
                "summary": "Technical setup lacks clear bias; monitor breakout catalysts.",
            })
        if "\"qualitative_summaries\"" in last_message:
            return json.dumps({
                "swot": {
                    "strengths": ["Diverse revenue streams", "Healthy balance sheet"],
                    "weaknesses": ["Margin pressure from input costs"],
                    "opportunities": ["Expansion in emerging markets"],
                    "threats": ["Competitive pricing pressure"],
                },
                "financial_health": "Stable",
                "overall_thesis": "neutral",
                "justification": "Solid fundamentals but limited near-term catalysts.",
            })
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
        if "\"options_chain\"" in last_message:
            return json.dumps({
                "strategy_name": "Long Call",
                "action": "BUY_TO_OPEN",
                "quantity": 1,
                "trade_legs": [
                    {
                        "contract_symbol": "FALLBACKCALL",
                        "type": "CALL",
                        "action": "BUY",
                        "strike_price": 100.0,
                        "expiration_date": "2025-01-17",
                        "quantity": 1,
                        "key_greeks_at_selection": {
                            "delta": 0.55,
                            "gamma": 0.04,
                            "theta": -0.02,
                            "vega": 0.09,
                            "impliedVolatility": 0.32,
                        },
                    }
                ],
                "notes": "Fallback trade suggestion in absence of model access.",
            })
        if "trade" in last_message.lower():
            return json.dumps({
                "strategy_name": "Call Debit Spread",
                "action": "BUY_TO_OPEN",
                "quantity": 1,
                "trade_legs": [],
                "notes": "Fallback trade suggestion in absence of model access.",
            })
        return "Model unavailable; defaulting to safe response."


def get_llm_client() -> LLMClient:
    return LLMClient()
