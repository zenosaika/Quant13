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
    """
    Parse LLM response and ensure it matches TradeThesis schema.

    TradeThesis schema:
    - winning_argument: str
    - conviction_level: str
    - summary: str
    - key_evidence: List[str]

    CRITICAL FIX: Extract JSON from code blocks and nested structures
    """
    import re

    # STEP 1: Try to extract JSON from markdown code blocks
    # Pattern: ```json\n{...}\n```
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if code_block_match:
        raw = code_block_match.group(1)

    try:
        data = json.loads(raw)

        # Validate it's a dict, not a list
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # STEP 2: Check if actual data is nested in 'summary' field (RBLX bug)
        # Sometimes LLM wraps the real thesis inside the summary as a JSON string
        if "summary" in data and isinstance(data["summary"], str):
            # Try to extract nested JSON from summary
            nested_match = re.search(r'\{[^{}]*"winning_argument"[^{}]*"conviction_level"[^{}]*\}', data["summary"], re.DOTALL)
            if nested_match:
                try:
                    nested_data = json.loads(nested_match.group(0))
                    if isinstance(nested_data, dict) and "winning_argument" in nested_data:
                        # The nested JSON is the actual thesis!
                        print(f"[DEBATE] WARNING: Found nested thesis in summary field. Using nested data.")
                        data = nested_data  # Replace with the correct nested data
                except json.JSONDecodeError:
                    pass  # Keep original data if nested parse fails

        # Ensure key_evidence is a list
        if "key_evidence" in data and not isinstance(data["key_evidence"], list):
            # If it's a string, wrap in list
            if isinstance(data["key_evidence"], str):
                data["key_evidence"] = [data["key_evidence"]]
            else:
                data["key_evidence"] = []

        # CRITICAL FIX: Do NOT use defaults if we have valid data
        # Only default if field is completely missing
        if "winning_argument" not in data:
            print("[DEBATE] ERROR: Missing winning_argument! Using fallback.")
            data["winning_argument"] = "Neutral"  # Changed from "Bullish" to "Neutral"
        if "conviction_level" not in data:
            print("[DEBATE] ERROR: Missing conviction_level! Using fallback.")
            data["conviction_level"] = "Low"
        if "summary" not in data:
            data["summary"] = raw.strip()
        if "key_evidence" not in data:
            data["key_evidence"] = []

        return data

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[DEBATE] ERROR: JSON parsing failed: {e}")
        # Fallback if JSON parsing fails
        return {
            "winning_argument": "Bullish",
            "conviction_level": "Low",
            "summary": raw.strip(),
            "key_evidence": [],
        }
