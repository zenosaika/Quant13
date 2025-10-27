from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.agents.base import Agent
from src.models.schemas import ArticleSentiment, SentimentReport
from src.tools.llm import get_llm_client


class SentimentAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = get_llm_client()

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        news_items: List[Dict[str, Any]] = state.get("news", [])
        recent_news = [item for item in news_items if _is_recent(_extract_timestamp(item))]
        payload_articles = [_normalize_article(item) for item in recent_news]
        if not payload_articles and news_items:
            payload_articles = [_normalize_article(news_items[0])]

        messages = [
            {"role": "system", "content": self.config["prompt"]},
            {"role": "user", "content": json.dumps({"ticker": state["ticker"], "articles": payload_articles})},
        ]
        raw_response = self.llm.chat(messages, temperature=0.3)
        parsed = _parse_sentiment_response(raw_response, payload_articles)

        return {
            "articles": parsed["articles"],
            "overall_sentiment_score": parsed.get("overall_sentiment_score", 0.0),
            "overall_summary": parsed.get("overall_summary", "No substantive sentiment identified."),
            "raw": raw_response,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> SentimentReport:
        return SentimentReport(
            ticker=state["ticker"],
            overall_sentiment_score=analysis["overall_sentiment_score"],
            overall_summary=analysis["overall_summary"],
            articles=[ArticleSentiment(**article) for article in analysis["articles"]],
        )


def _is_recent(timestamp: Any, lookback_days: int = 7) -> bool:
    if not timestamp:
        return False
    try:
        if isinstance(timestamp, (int, float)):
            ts = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        elif isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        elif isinstance(timestamp, datetime):
            ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        else:
            return False
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - ts).days <= lookback_days


def _normalize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    timestamp = _extract_timestamp(item)
    published_at = None
    if timestamp:
        try:
            if isinstance(timestamp, (int, float)):
                published_at = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
            elif isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                published_at = ts.isoformat()
            elif isinstance(timestamp, datetime):
                ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
                published_at = ts.isoformat()
        except (TypeError, ValueError):
            published_at = None
    return {
        "title": item.get("title") or item.get("headline") or content.get("title") or "",
        "publisher": item.get("publisher") or content.get("publisher") or content.get("provider"),
        "link": item.get("link") or content.get("canonicalUrl") or content.get("link"),
        "published_at": published_at,
        "summary": item.get("summary") or content.get("summary"),
    }


def _extract_timestamp(item: Dict[str, Any]) -> Any:
    if item.get("providerPublishTime"):
        return item.get("providerPublishTime")
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    if content.get("pubDate"):
        return content.get("pubDate")
    if content.get("displayTime"):
        return content.get("displayTime")
    return None


def _parse_sentiment_response(raw: str, fallback_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    articles_payload = []
    for article in data.get("articles", []):
        if not isinstance(article, dict):
            continue
        title = article.get("title")
        if not title:
            continue
        sentiment_score = article.get("sentiment_score")
        try:
            sentiment_score = float(sentiment_score)
        except (TypeError, ValueError):
            sentiment_score = 0.0
        articles_payload.append({
            "title": title,
            "publisher": article.get("publisher"),
            "link": article.get("link"),
            "published_at": article.get("published_at"),
            "sentiment_score": sentiment_score,
            "rationale": article.get("rationale"),
        })

    if not articles_payload:
        for article in fallback_articles:
            articles_payload.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "sentiment_score": 0.0,
                "rationale": "Insufficient data; defaulting to neutral sentiment.",
            })

    overall_score = data.get("overall_sentiment_score")
    try:
        overall_score = float(overall_score)
    except (TypeError, ValueError):
        overall_score = sum(a["sentiment_score"] for a in articles_payload) / len(articles_payload) if articles_payload else 0.0

    overall_summary = data.get("overall_summary") or "Sentiment evaluated as neutral given limited signals."

    return {
        "articles": articles_payload,
        "overall_sentiment_score": overall_score,
        "overall_summary": overall_summary,
    }
