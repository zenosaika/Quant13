from __future__ import annotations

import json
from typing import Any, Dict

from src.models.schemas import TradeProposal, TradeThesis, VolatilityReport
from src.tools.llm import get_llm_client


class TraderAgent:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.llm = get_llm_client()

    def propose_trade(self, thesis: TradeThesis, volatility_report: VolatilityReport) -> TradeProposal:
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps({
                "thesis": thesis.model_dump(),
                "volatility": volatility_report.model_dump(),
                }),
            },
        ]
        raw = self.llm.chat(messages)
        data = _safe_json_loads(raw)
        return TradeProposal(**data)


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "strategy": "Covered Call",
            "direction": "Neutral",
            "expiration": "30D",
            "strikes": [],
            "notes": raw.strip(),
        }
