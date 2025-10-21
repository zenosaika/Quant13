from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agents.base import Agent
from src.models.schemas import NewsHeadline, SentimentReport
from src.tools.knowledge_graph import knowledge_graph_alerter


class SentimentAgent(Agent):
    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        news_items: List[Dict[str, Any]] = state.get("news", [])
        positive = {word.lower() for word in self.config.get("positive_keywords", [])}
        negative = {word.lower() for word in self.config.get("negative_keywords", [])}

        headlines: List[NewsHeadline] = []
        aggregate_score = 0.0
        kg_insights: List[Dict[str, str]] = []

        for item in news_items:
            raw_headline = item.get("title") or item.get("headline") or ""
            summary = item.get("summary")
            text = f"{raw_headline} {summary or ''}".lower()
            score = _score_text(text, positive, negative)
            aggregate_score += score
            headlines.append(
                NewsHeadline(
                    headline=raw_headline,
                    summary=summary,
                    link=item.get("link"),
                    published_at=item.get("providerPublishTime"),
                    score=score,
                )
            )
            if abs(score) >= 0.6 and not kg_insights:
                kg_insights.append(knowledge_graph_alerter(state["ticker"]))

        overall = aggregate_score / len(headlines) if headlines else 0.0
        return {
            "overall_sentiment_score": round(overall, 3),
            "headlines": headlines,
            "kg_insights": kg_insights or None,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> SentimentReport:
        return SentimentReport(
            ticker=state["ticker"],
            overall_sentiment_score=analysis["overall_sentiment_score"],
            key_headlines=analysis["headlines"],
            kg_derived_insights=analysis["kg_insights"],
        )


def _score_text(text: str, positive: set[str], negative: set[str]) -> float:
    if not text:
        return 0.0
    tokens = re.findall(r"[a-zA-Z]+", text)
    if not tokens:
        return 0.0
    pos_hits = sum(token in positive for token in tokens)
    neg_hits = sum(token in negative for token in tokens)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / total
