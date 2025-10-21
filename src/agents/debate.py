from __future__ import annotations

import json
from typing import Any, Dict, List

from src.models.schemas import DebateArgument, TradeThesis
from src.tools.llm import get_llm_client


class ResearcherAgent:
    def __init__(self, prompt: str, stance: str) -> None:
        self.prompt = prompt
        self.stance = stance
        self.llm = get_llm_client()

    def run(self, reports: Dict[str, Any]) -> DebateArgument:
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps({"stance": self.stance, "reports": reports}),
            },
        ]
        response = self.llm.chat(messages)
        return DebateArgument(stance=self.stance, argument=response)


class ModeratorAgent:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.llm = get_llm_client()

    def run(self, debate_transcript: List[DebateArgument], reports: Dict[str, Any]) -> TradeThesis:
        transcript = [arg.model_dump() for arg in debate_transcript]
        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "transcript": transcript,
                    "reports": reports,
                }),
            },
        ]
        raw = self.llm.chat(messages)
        thesis_data = _safe_json_loads(raw)
        return TradeThesis(**thesis_data)


class DebateOrchestrator:
    def __init__(self, config: Dict[str, str]) -> None:
        self.bullish = ResearcherAgent(config["bullish_prompt"], stance="Bullish")
        self.bearish = ResearcherAgent(config["bearish_prompt"], stance="Bearish")
        self.moderator = ModeratorAgent(config["moderator_prompt"])

    def conduct_debate(self, reports: Dict[str, Any]) -> TradeThesis:
        arguments = [
            self.bullish.run(reports),
            self.bearish.run(reports),
        ]
        thesis = self.moderator.run(arguments, reports)
        return thesis


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "winning_argument": "Bullish",
            "conviction_level": "Low",
            "summary": raw.strip(),
            "key_evidence": [],
        }
