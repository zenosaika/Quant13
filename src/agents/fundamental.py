from __future__ import annotations

from typing import Any, Dict

from src.agents.base import Agent
from src.models.schemas import FundamentalReport
from src.tools.knowledge_graph import knowledge_graph_querier


class FundamentalAnalyst(Agent):
    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = state.get("company_overview", {})
        ratios = {
            "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "gross_margin": _format_percentage(info.get("grossMargins")),
            "debt_to_equity": round(info.get("debtToEquity"), 2) if info.get("debtToEquity") else None,
            "free_cash_flow": info.get("freeCashflow"),
        }
        kg_swot = knowledge_graph_querier(state["ticker"])
        return {"financial_health": ratios, "kg_swot": kg_swot}

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> FundamentalReport:
        return FundamentalReport(
            ticker=state["ticker"],
            financial_health=analysis["financial_health"],
            kg_derived_swot=analysis["kg_swot"],
        )


def _format_percentage(value: Any) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.1f}%"
